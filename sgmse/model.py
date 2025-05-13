import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import numpy as np
import random
import wandb

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model, evaluate_model_ctm, evaluate_model_ctm_nstep
from sgmse.util.other import si_sdr_loss, pad_spec
#from thop import profile
import os
from os import makedirs
from os.path import join, dirname
from soundfile import write
from torch_pesq import PesqLoss
from timm.scheduler.cosine_lr import CosineLRScheduler
from asteroid.losses import singlesrc_neg_sisdr
import torch.distributed as dist


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03,
        num_eval_files=19, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.sigma_data = 0.1

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x, y, _, _ = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y)
        err = score * sigmas + z
        loss = self._loss(err)
        return loss

    def training_step(self, batch, batch_idx):
        try:
            loss = self._step(batch, batch_idx)
            self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)            
            return loss
        except Exception as e:
            print(f'Error in batch {batch_idx}: {e}')
            raise e

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def _c_in(self, t):
        return 1.0            
    
    def _c_out(self, t):
        return 1.0            
    
    def _c_skip(self, t):
        return 0.0            

    def forward(self, x, y, t):
        if self.backbone == "ncsnpp_v2" or self.backbone == "ncsnpp_48k_v2":
            F = self.dnn(self._c_in(t) * x, self._c_in(t) * y, t)
            x_hat = self._c_skip(t) * x + self._c_out(t) * F
            return x_hat
        else:
            # Concatenate y as an extra channel
            dnn_input = torch.cat([x, y], dim=1)
            
            # the minus is most likely unimportant here - taken from Song's repo
            score = -self.dnn(dnn_input, t)

        if 0:
            macs, params = profile(self.dnn, (dnn_input, t))
            gflops = 2. * macs / 1e9
            params_mb = params / (1024 * 1024)
            print(f'{gflops} [GFLOPs]')
            print(f'{params_mb} [MB]')

        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)
    
    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=16000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        elif sampler_type == "ode":
            sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat


class CTModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--neval_files", type=int, default=24, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")        
        parser.add_argument("--opt", type=str, default="adam", help="Optimizer")
        parser.add_argument("--ctm_lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--scheduler", type=str, default="None", help="Scheduler")
        parser.add_argument("--ctm_rec_w", type=float, default=0.0, help="CTM Reconstruction loss weight")
        parser.add_argument("--ctm_psq_w", type=float, default=0.0, help="CTM PESQ loss weight")
        parser.add_argument("--dsm_rec_w", type=float, default=0.0, help="DSM Reconstruction loss weight")
        parser.add_argument("--dsm_psq_w", type=float, default=0.0, help="DSM PESQ loss weight")
        parser.add_argument("--cm", action='store_true', help="Enable Consistency Model")
        return parser

    def __init__(self, backbone, sde, ckpt_dir=None, ctm_lr=1e-4, opt='adam', scheduler='None', cm=False, neval_files=24, ctm_rec_w=0.0, ctm_psq_w=0.0, dsm_rec_w=0.0, dsm_psq_w=0.0, data_module_cls=None, teacher=None, **kwargs):
        super().__init__()
        self.cm = cm
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        self.t_min_infer = 1e-4
        self.t_min_train = 3e-2
        self.t_max_train = 1.0
        self.sigma_data = 0.1
        self.step_max = 40
        self.rho = 7
        self.ckpt_dir = ckpt_dir
        self.save_hyperparameters(ignore=['no_wandb', 'teacher'])
        self.opt_params_list = list(self.dnn.parameters())
        # Store hyperparams and save them
        self.opt = opt
        self.lr = ctm_lr
        if scheduler == 'None':
            self.scheduler = None
        else:
            self.scheduler = scheduler
        self.loss_type = 'mse'
        self.lambda_ctm_rec = ctm_rec_w
        self.lambda_ctm_psq = ctm_psq_w
        self.lambda_dsm_rec = dsm_rec_w
        self.lambda_dsm_psq = dsm_psq_w
        self.num_eval_files = neval_files
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.lambda_dsm = 0.0
        if self.backbone == 'ncsnpp-ctm_48k' or self.backbone == 'ncsnpp-ctm_48k_v2':
            self.sr = 48000
        else:
            self.sr = 16000
        # PESQ loss
        self.pesq_loss = PesqLoss(1.0, sample_rate=self.sr).eval()
        for param in self.pesq_loss.parameters():
            param.requires_grad = False
        # Teacher
        self.teacher = teacher
        if self.teacher != None:
            self.teacher.requires_grad_(False)
            self.teacher.eval()
        # Target
        if backbone == 'ncsnpp-ctm_v2' or backbone == 'ncsnpp-ctm_48k_v2':
            self.dnn_tgt = dnn_cls(**kwargs)
        else:
            self.dnn_tgt = dnn_cls(scale_by_sigma=False, **kwargs)
        self.target_params_list = list(self.dnn_tgt.parameters())
        for param in self.target_params_list:
            param.requires_grad = False

    def on_train_start(self):
        if self.cm:
            print(f'\n[INF]: Consistency Model Training')
        else:
            print(f'\n[INF]: Consistency Trajectory Model Training')
        print(f'\n[INF]: Start from {self.current_epoch} epoch / {self.trainer.global_step} step')
        if self.trainer.global_step == 0:
            # Copy student parameters to target parameters
            for dst, src in zip(self.dnn_tgt.parameters(), self.dnn.parameters()):
                dst.data.copy_(src.data)
            print(f'[INF]: Initialized target model with student model')
        if self.scheduler != None:
            self.scheduler.step(epoch=self.trainer.current_epoch)

    def configure_optimizers(self):
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.opt_params_list, lr=self.lr)
            print(f'Optimizer: Adam (lr={self.lr:e})')
        elif self.opt == 'radam':
            optimizer = torch.optim.RAdam(self.opt_params_list, lr=self.lr)
            print(f'Optimizer: RAdam (lr={self.lr:e})')
        elif self.opt == 'adamw':
            optimizer = torch.optim.AdamW(self.opt_params_list, lr=self.lr)
            print(f'Optimizer: AdamW (lr={self.lr:e})')
        else:
            assert False, "Define optimizer!"
        if self.scheduler == 'wcos':
            self.scheduler = CosineLRScheduler(optimizer, t_initial=self.trainer.max_epochs, warmup_t=5, lr_min=5e-6, warmup_lr_init=1e-5, warmup_prefix=False)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        else:
            self.scheduler = None
            return optimizer
        
    def lr_scheduler_step(self, scheduler, optimizer_idx, metrics=None):
        scheduler.step(epoch=self.trainer.current_epoch)
        print(f"lr: {scheduler.optimizer.param_groups[0]['lr']:.4e}")
        
    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        with torch.no_grad():
            # Update sg(th) <- stopgrad(mu*sg(th)+(1-mu)*th)
            self._update_sg(self.target_params_list, self.opt_params_list, rate=0.999)

    @torch.no_grad()
    def _update_sg(self, dst_params, src_params, rate=0.99):
        for dst, src in zip(dst_params, src_params):
            dst.detach().mul_(rate).add_(src, alpha=1 - rate)

    def _calc_lambda_dsm(self, loss_ctm, loss_dsm, last_layers_weight):
        loss_ctm_grad = torch.autograd.grad(loss_ctm, last_layers_weight, retain_graph=True, allow_unused=False)[0]
        loss_dsm_grad = torch.autograd.grad(loss_dsm, last_layers_weight, retain_graph=True)[0]
        lambda_dsm = torch.norm(loss_ctm_grad) / (torch.norm(loss_dsm_grad) + 1e-8)
        lambda_dsm = torch.clamp(lambda_dsm, 0.0, 1e3).detach()
        return lambda_dsm

    def g_th(self, x_t, t, s, y, model):
        sigma_data = self.sigma_data
        if self.backbone == 'ncsnpp-ctm_v2':
            c_in   = torch.full((y.shape[0],), 1.0, device=y.device)
            c_skip = torch.full((y.shape[0],), 0.0, device=y.device)
            c_out  = torch.full((y.shape[0],), 1.0, device=y.device)
        elif self.backbone == 'ncsnpp-ctm_48k_v2':
            sigma = self.sde._std(t)
            c_in   = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
            c_skip = (sigma_data ** 2) / (sigma ** 2 + sigma_data ** 2)
            c_out  = (sigma * sigma_data) / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        elif self.backbone == 'ncsnpp-ctm_48k':
            c_in = 1.0 / torch.sqrt(t ** 2 + sigma_data ** 2)
            c_skip = (sigma_data ** 2) / (t ** 2 + sigma_data ** 2)
            c_out = (t * sigma_data) / torch.sqrt(t ** 2 + sigma_data ** 2)
        else:
            c_in   = torch.full((y.shape[0],), 1.0, device=y.device)
            c_skip = torch.full((y.shape[0],), 0.0, device=y.device)
            c_out  = torch.full((y.shape[0],), 1.0, device=y.device)
        c_in   = c_in[:, None, None, None]
        c_skip = c_skip[:, None, None, None]
        c_out  = c_out[:, None, None, None]
        out    = c_skip * x_t + c_out * self(c_in * x_t, t, s, c_in * y, model)
        return out

    def G_th(self, x_t, t, s, y, model):
        # G_th(x_t, t, s) = s/t*x_t + (1-s/t)*g_th(x_t, t, s)
        if self.backbone == 'ncsnpp-ctm_48k':
            if x_t == None:
                x_t = self.sde.prior_sampling(y.shape, y).to(y.device)
        g_th_out = self.g_th(x_t, t, s, y, model)
        t = t[:, None, None, None]
        s = s[:, None, None, None]
        out = (s / t) * x_t + (1 - (s / t)) * g_th_out
        return out

    def _wout(self, data, normfac, tgt_len, fname):
        out = self.to_audio(data.squeeze(), tgt_len)
        out = out * normfac
        makedirs(dirname(join('eval/output', fname)), exist_ok=True)
        write(join('eval/output', fname), out.cpu().numpy(), self.sr)

    def get_step_idx(self):
        p = np.array([i for i in range(1, self.step_max)])
        p = p / sum(p) # probability
        step_idx = np.random.choice([i + 1 for i in range(len(p))], size=1, p=p)[0]
        return step_idx

    def sample_t_idx(self, batch_size, device, step_idx=1):
        w = np.ones([self.step_max - step_idx])
        p = w / np.sum(w)
        t_idx_np = np.random.choice(len(p), size=(batch_size,), p=p)
        t_idx = torch.from_numpy(t_idx_np).long().to(device)
        return t_idx

    def sample_s_idx(self, batch_size, device, idx, step_idx=1):
        s_idx_np = np.random.randint(low=(idx + step_idx).cpu().detach().numpy(), high=self.step_max, size=(batch_size,), dtype=int)
        s_idx = torch.from_numpy(s_idx_np).to(device)
        return s_idx

    def get_t(self, input, sigma_min):
        sigma_max = self.t_max_train
        sigmas = (sigma_max ** (1 / self.rho) + input * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        return sigmas

    def get_dsmt(self, batch_size, device):
        sigmas1 = torch.rand(batch_size // 2, device=device) * (self.t_max_train - self.t_min_train) + self.t_min_train # sigmas1: [t_min_train, t_max_train)
        t2 = torch.rand(batch_size - batch_size // 2, device=device) * 0.7 # t2: [0, 0.7)
        sigmas2 = self.get_t(t2, self.t_min_train) # sigmas2: (get_t(0.7), t_max_train]
        sigmas = torch.cat((sigmas1, sigmas2)).view(-1) # sigmas: [t_min_train, t_max_train]
        return sigmas

    def _step(self, batch, batch_idx):
        result = {}
        x_0_ref, y, normfac, ref_len = batch
        batch_size = y.shape[0]
        sde_teacher = self.teacher.sde.copy()
        t_max = torch.full((batch_size,), self.t_max_train, device=y.device)
        t_min = torch.full((batch_size,), self.t_min_train, device=y.device)
        if self.cm: # SBCM (u=t-dt, s=0)
            step_idx = self.get_step_idx()
            t_idx = self.sample_t_idx(batch_size, y.device, step_idx)
            s_idx = torch.full((batch_size,), self.step_max - 1, device=y.device)
            t = self.get_t(t_idx       / (self.step_max - 1), self.t_min_train)
            u = self.get_t((t_idx + 1) / (self.step_max - 1), self.t_min_train)
            s = self.get_t(s_idx       / (self.step_max - 1), self.t_min_train)
            # using same step_idx for each t, linear step for teacher
            indices = torch.linspace(0, 1, 2, device=y.device).unsqueeze(0)
            timesteps_t_u = t.unsqueeze(1) + (u - t).unsqueeze(1) * indices # [t, ..., u], "step_idx" steps
        else: # SBCTM
            step_idx = self.get_step_idx()
            t_idx = self.sample_t_idx(batch_size, y.device, step_idx)
            s_idx = self.sample_s_idx(batch_size, y.device, t_idx, step_idx)
            t = self.get_t(t_idx              / (self.step_max - 1), self.t_min_train)
            u = self.get_t((t_idx + step_idx) / (self.step_max - 1), self.t_min_train)
            s = self.get_t(s_idx              / (self.step_max - 1), self.t_min_train)
            #assert torch.all((t_min <= s) & (s <= u) & (u < t) & (t <= t_max)), f"t_min:{t_min}, s:{s}, u:{u}, t:{t}, t_max:{t_max}"
            # using same step_idx for each t, linear step for teacher
            indices = torch.linspace(0, 1, step_idx + 1, device=y.device).unsqueeze(0)
            timesteps_t_u = t.unsqueeze(1) + (u - t).unsqueeze(1) * indices # [t, ..., u], "step_idx" steps

        # sampling x_t from marginal probability by SDE (== teacher's forward SDE)
        mean, std = self.sde.marginal_prob(x_0_ref, y, t)
        z = torch.randn_like(x_0_ref)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        # override x_t = y for t == T (for ODE)
        mask = (t == self.t_max_train)
        x_t[mask] = y[mask]

        with torch.no_grad():
            # t->u: TeacherSolver(x_t, t, u)
            if self.backbone == "ncsnpp-ctm_v2" or self.backbone == "ncsnpp-ctm_48k_v2":
                x_u_teacher = sampling.get_sb_sampler_ode(sde_teacher, self.teacher, x_t.clone(), y, timesteps_t_u)
            else:
                x_u_teacher, _ = sampling.step_denoise(x_t.clone(), y, sde_teacher, self.teacher, timesteps_t_u)
            # u->s: G_sg(th)(x_u_teacher, u, s)
            x_s_tgt = self.G_th(x_u_teacher, u, s, y, self.dnn_tgt)
            # s->0: G_sg(th)(x_s_tgt, s, 0)
            x_0_tgt = self.G_th(x_s_tgt, s, t_min, y, self.dnn_tgt)
        # t->s: G_th(x_t, t, s)
        x_s_est = self.G_th(x_t, t, s, y, self.dnn)
        # s->0: G_sg(th)(x_s_est, s, 0)
        x_0_est = self.G_th(x_s_est, s, t_min, y, self.dnn_tgt)

        # frequency domain -> time domain
        B, C, F, T = x_0_ref.shape
        if self.cm == False:
            tgt_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_0_ref_td = self.to_audio(x_0_ref.squeeze(), tgt_len)
            x_0_est_td = self.to_audio(x_0_est.squeeze(), tgt_len)
            x_0_tgt_td = self.to_audio(x_0_tgt.detach().squeeze(), tgt_len)

        # [CTM] time-frequency domain loss
        loss_tfd_ctm = x_0_est - x_0_tgt.detach()
        loss_tfd_ctm = (1.0 / (F * T)) * torch.square(torch.abs(loss_tfd_ctm))
        loss_tfd_ctm = torch.mean(0.5 * torch.sum(loss_tfd_ctm.reshape(loss_tfd_ctm.shape[0], -1), dim=-1))

        if self.cm: # SBCM
            loss = loss_tfd_ctm
            loss_dsm = 0
            loss_ref = 0
            self.lambda_dsm = 0
            self.lambda_ref = 0
            loss_ctm = 0
            loss_dsm = 0
            loss_ref = 0
            loss_tfd_ctm = 0
            loss_rec_ctm = 0
            loss_psq_ctm = 0
            loss_tfd_dsm = 0
            loss_rec_dsm = 0
            loss_psq_dsm = 0
            loss_tfd_ref = 0
            loss_rec_ref = 0
            loss_psq_ref = 0
            dsm_t = torch.full((batch_size,), 0, device=y.device)
        else: # SBCTM
            # [CTM] reconstruction loss
            loss_rec_ctm = (1.0 / tgt_len) * torch.abs(x_0_est_td - x_0_tgt_td)
            loss_rec_ctm = torch.mean(0.5 * torch.sum(loss_rec_ctm.reshape(loss_rec_ctm.shape[0], -1), dim=-1))
            # [CTM] PESQ loss
            loss_psq_ctm = self.pesq_loss(x_0_est_td, x_0_tgt_td)
            loss_psq_ctm = torch.mean(loss_psq_ctm)
            # [CTM] loss
            loss_ctm = loss_tfd_ctm + loss_rec_ctm * self.lambda_ctm_rec + loss_psq_ctm * self.lambda_ctm_psq

            # sampling t for DSM Loss
            dsm_t = self.get_dsmt(batch_size, y.device)
            # sampling x_t from marginal probability by SDE (== teacher's forward SDE)
            mean, std = self.sde.marginal_prob(x_0_ref, y, dsm_t)
            z = torch.randn_like(x_0_ref)
            sigma = std[:, None, None, None]
            x_t = mean + sigma * z
            # override x_t = y for t == T (for ODE)
            mask = (dsm_t == self.t_max_train)
            x_t[mask] = y[mask]
            dsm = self.g_th(x_t, dsm_t, dsm_t, y, self.dnn)
            dsm_td = self.to_audio(dsm.squeeze(), tgt_len)
            # [DSM] time-frequency domain loss
            loss_tfd_dsm = x_0_ref - dsm
            loss_tfd_dsm = (1.0 / (F * T)) * torch.square(torch.abs(loss_tfd_dsm))
            loss_tfd_dsm = torch.mean(0.5 * torch.sum(loss_tfd_dsm.reshape(loss_tfd_dsm.shape[0], -1), dim=-1))
            if 1:
                # [DSM] reconstruction loss
                loss_rec_dsm = (1.0 / tgt_len) * torch.abs(x_0_ref_td - dsm_td)
                loss_rec_dsm = torch.mean(0.5 * torch.sum(loss_rec_dsm.reshape(loss_rec_dsm.shape[0], -1), dim=-1))
                # [DSM] PESQ loss
                loss_psq_dsm = self.pesq_loss(x_0_ref_td, dsm_td)
                loss_psq_dsm = torch.mean(loss_psq_dsm)
                # [DSM] loss
                loss_dsm = loss_tfd_dsm + loss_rec_dsm * self.lambda_dsm_rec + loss_psq_dsm * self.lambda_dsm_psq
                if self.training: # when validation, use latest saved one
                    self.lambda_dsm = self._calc_lambda_dsm(loss_ctm.mean(), loss_dsm.mean(), last_layers_weight=self.dnn.output_layer.weight)
            else:
                loss_rec_dsm = 0
                loss_psq_dsm = 0
                loss_dsm = loss_tfd_dsm
                self.lambda_dsm = 0

            # total Loss (CTM Loss + DSM Loss)
            loss = loss_ctm + loss_dsm * self.lambda_dsm

        if 0:
            with torch.no_grad():
                timesteps = torch.linspace(self.t_max_train - self.gamma, self.t_min_train, 8, device=y.device).unsqueeze(0).expand(batch_size, -1)
                x_t = sampling.get_sb_sampler_sde_1step(sde_teacher, self.teacher, y)
                x_t = sampling.get_sb_sampler_ode(sde_teacher, self.teacher, x_t, y, timesteps)
                for b in range(batch_size):
                    self._wout(x_t[b], normfac[b], ref_len[b], f'out{b}.wav')

        result['Loss']         = loss
        result['Loss-CTM']     = loss_ctm
        result['Loss-DSM']     = loss_dsm * self.lambda_dsm
        result['Loss-TFD-CTM'] = loss_tfd_ctm
        result['Loss-REC-CTM'] = loss_rec_ctm * self.lambda_ctm_rec
        result['Loss-PSQ-CTM'] = loss_psq_ctm * self.lambda_ctm_psq
        result['Loss-TFD-DSM'] = loss_tfd_dsm
        result['Loss-REC-DSM'] = loss_rec_dsm * self.lambda_dsm_rec
        result['Loss-PSQ-DSM'] = loss_psq_dsm * self.lambda_dsm_psq
        result['Lambda-DSM']   = self.lambda_dsm
        result['t']            = t[0].item()
        result['s']            = s[0].item()
        result['u']            = u[0].item()
        result['t-DSM']        = dsm_t[0].item()
        return result
   
    def training_step(self, batch, batch_idx):
        try:
            res = self._step(batch, batch_idx)
            self.log('Train Loss',         res['Loss'],         on_step=True,  on_epoch=True,  sync_dist=True)
            self.log('Train Loss-CTM',     res['Loss-CTM'],     on_step=True,  on_epoch=True,  sync_dist=True)
            self.log('Train Loss-DSM',     res['Loss-DSM'],     on_step=True,  on_epoch=True,  sync_dist=True)
            self.log('Train Loss-TFD-CTM', res['Loss-TFD-CTM'], on_step=False, on_epoch=True,  sync_dist=True)
            self.log('Train Loss-REC-CTM', res['Loss-REC-CTM'], on_step=False, on_epoch=True,  sync_dist=True)
            self.log('Train Loss-PSQ-CTM', res['Loss-PSQ-CTM'], on_step=False, on_epoch=True,  sync_dist=True)
            self.log('Train Loss-TFD-DSM', res['Loss-TFD-DSM'], on_step=False, on_epoch=True,  sync_dist=True)
            self.log('Train Loss-REC-DSM', res['Loss-REC-DSM'], on_step=False, on_epoch=True,  sync_dist=True)
            self.log('Train Loss-PSQ-DSM', res['Loss-PSQ-DSM'], on_step=False, on_epoch=True,  sync_dist=True)
            self.log('Train Lambda-DSM',   res['Lambda-DSM'],   on_step=True,  on_epoch=False, sync_dist=True)
            #self.log('t',                  res['t'],            on_step=True,  on_epoch=False, sync_dist=False)
            #self.log('s',                  res['s'],            on_step=True,  on_epoch=False, sync_dist=False)
            #self.log('u',                  res['u'],            on_step=True,  on_epoch=False, sync_dist=False)
            #self.log('dsmt',               res['t-DSM'],        on_step=True,  on_epoch=False, sync_dist=False)
            return res['Loss']
        except Exception as e:
            print(f'Error in batch {batch_idx}: {e}')
            raise e

    def validation_step(self, batch, batch_idx):
        res = self._step(batch, batch_idx)
        self.log('Valid Loss',         res['Loss'],         on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-CTM',     res['Loss-CTM'],     on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-DSM',     res['Loss-DSM'],     on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-TFD-CTM', res['Loss-TFD-CTM'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-REC-CTM', res['Loss-REC-CTM'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-PSQ-CTM', res['Loss-PSQ-CTM'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-TFD-DSM', res['Loss-TFD-DSM'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-REC-DSM', res['Loss-REC-DSM'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('Valid Loss-PSQ-DSM', res['Loss-PSQ-DSM'], on_step=False, on_epoch=True, sync_dist=True)
        if batch_idx == 0 and self.num_eval_files != 0:
            nfe_list = [1, 2, 4, 8]
            sdr, psq = evaluate_model_ctm_nstep(self, self.num_eval_files, nfe_list, self.dnn)
            sdr_tgt, psq_tgt = evaluate_model_ctm_nstep(self, self.num_eval_files, nfe_list, self.dnn_tgt)
            for n in range(len(nfe_list)):
                self.log(f'SISDR_{nfe_list[n]}',     sdr[n],     on_step=False, on_epoch=True, sync_dist=True)
                self.log(f'PESQ_{nfe_list[n]}',      psq[n],     on_step=False, on_epoch=True, sync_dist=True)
                self.log(f'SISDR_TGT_{nfe_list[n]}', sdr_tgt[n], on_step=False, on_epoch=True, sync_dist=True)
                self.log(f'PESQ_TGT_{nfe_list[n]}',  psq_tgt[n], on_step=False, on_epoch=True, sync_dist=True)
        return res['Loss']

    def forward(self, x_t, t, s, y, model):
        if self.backbone == "ncsnpp_48k_v2" or self.backbone == "ncsnpp-ctm_v2" or self.backbone == "ncsnpp-ctm_48k_v2":
            out = model(x_t, y, t, s)
        else:
            # Concatenate y as an extra channel
            model_in = torch.cat([x_t, y], dim=1)
            out = model(model_in, t, s)
        return out

    def get_ctm_sample(self, y, steps, dnn):
        with torch.no_grad():
            indices = torch.linspace(0, 1, steps + 1, device=y.device)
            timesteps = self.get_t(indices, self.t_min_infer)
            if self.backbone == "ncsnpp-ctm_v2" or self.backbone == "ncsnpp-ctm_48k_v2":
                x_t = y.clone()
            else:
                x_t = None

            if 0: # gamma-sampilng
                if self.cm:
                    gamma = 1.0
                else:
                    gamma = 0.0
                for i in range(len(timesteps) - 1):
                    t = torch.unsqueeze(timesteps[i], 0)
                    s = torch.unsqueeze(torch.sqrt(1.0 - torch.square(gamma)) * timesteps[i + 1], 0)
                    x_t = self.G_th(x_t, t, s, dnn)
                    if i < (len(timesteps) - 2):
                        t = torch.unsqueeze(timesteps[i + 1], 0)
                        mean, std = self.sde.marginal_prob(x_t, y, t)
                        z = torch.randn_like(x_t)
                        sigma = std[:, None, None, None]
                        x_t = mean + sigma * z
            else:
                if self.cm: # SBCM
                    for i in range(len(timesteps) - 1):
                        t = torch.unsqueeze(timesteps[i], 0)
                        s = torch.full((x_t.shape[0],), self.t_min_infer, device=y.device)
                        x_t = self.G_th(x_t, t, s, y, dnn)
                        if i < (len(timesteps) - 2):
                            t = torch.unsqueeze(timesteps[i + 1], 0)
                            mean, std = self.sde.marginal_prob(x_t, y, t)
                            z = torch.randn_like(x_t)
                            sigma = std[:, None, None, None]
                            x_t = mean + sigma * z
                else: # SBCTM
                    for i in range(len(timesteps) - 1):
                        t = torch.unsqueeze(timesteps[i    ], 0)
                        s = torch.unsqueeze(timesteps[i + 1], 0)
                        x_t = self.G_th(x_t, t, s, y, dnn)
            return x_t

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
