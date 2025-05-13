import glob
import warnings
import torch
from os import makedirs
from os.path import join, dirname
from argparse import ArgumentParser
from soundfile import write
from torchaudio import load
from tqdm import tqdm
from thop import clever_format
from thop import profile
import numpy as np
import random
import time

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
set_torch_cuda_arch_list()

from sgmse.model import ScoreModel
from sgmse.model import CTModel
from sgmse.util.other import pad_spec
from sgmse.sdes import SDERegistry
from sgmse import sampling

## Wrapper Function for Profiling
class SgmseWrapper(torch.nn.Module):
    def __init__(self, model, args):
        super(SgmseWrapper, self).__init__()
        self.model = model
        self.args = args
    
    def forward(self, input):
        if 0:
            sampler = self.model.get_pc_sampler('reverse_diffusion', self.args.corrector, input, N=self.args.N, corrector_steps=self.args.corrector_steps, snr=self.args.snr)
            sample = sampler()
        else:
            sample = self.model.get_ctm_sample(input, self.args.N, self.dnn)
        return sample

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    set_seed(42)
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    parser.add_argument("--ckpt_path", type=str,  help='Path to model checkpoint.')
    parser.add_argument("--sampler_type", type=str, default="pc", help="Sampler type for the PC sampler.")    
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--ctm", type=str, default='True')
    parser.add_argument("--tgtm", type=str, default='True')
    args = parser.parse_args()

    if args.ctm == 'True':
        model = CTModel.load_from_checkpoint(args.ckpt_path, map_location='cuda', strict=False)
        model.eval()
    else:
        # Load score model    
        model = ScoreModel.load_from_checkpoint(args.ckpt_path, map_location='cuda', strict=False)
        model.eval(no_ema=False)
    model.cuda()

    with torch.no_grad():
        # Get list of noisy files
        noisy_files = []
        noisy_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
        noisy_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))

        # Check if the model is trained on 48 kHz data
        if model.backbone == 'ncsnpp' or model.backbone == 'ncsnpp_v2' or model.backbone == 'ncsnpp-ctm_v2':
            sr = 16000
            pad_mode = "reflection"
        elif model.backbone == 'ncsnpp_48k' or model.backbone == 'ncsnpp_48k_v2' or model.backbone == 'ncsnpp-ctm_48k' or model.backbone == 'ncsnpp-ctm_48k_v2':
            sr = 48000
            pad_mode = "reflection"
        else:
            sr = 16000
            pad_mode = "zero_pad"

        # Enhance files
        for noisy_file in tqdm(noisy_files):
            filename = noisy_file.split('/')[-1]
            filename = noisy_file.replace(args.test_dir, "")[1:] # Remove the first character which is a slash
            
            # Load wav
            y, _ = load(noisy_file) 
            T_orig = y.size(1)   

            # Normalize
            norm_factor = y.abs().max()
            y = y / norm_factor

            # Prepare DNN input
            Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
            Y = pad_spec(Y, mode=pad_mode)
            
            if 0:
                n_warmup = 5
                n_repeat = 10
                # warmup
                for _ in range(n_warmup):
                    if args.ctm == 'True':
                        if args.tgtm == 'True':
                            sample = model.get_ctm_sample(Y.cuda(), args.N, model.dnn_tgt)
                        else:
                            sample = model.get_ctm_sample(Y.cuda(), args.N, model.dnn)
                torch.cuda.synchronize()
                # main
                start_time = time.time()
                for _ in range(n_repeat):
                    if args.ctm == 'True':
                        if args.tgtm == 'True':
                            sample = model.get_ctm_sample(Y.cuda(), args.N, model.dnn_tgt)
                        else:
                            sample = model.get_ctm_sample(Y.cuda(), args.N, model.dnn)
                torch.cuda.synchronize()

                end_time = time.time()
                total_time = end_time - start_time
                avg_time = total_time / n_repeat
                print(f'Average inference time: {avg_time:.4f} sec')
            else:
                # Reverse sampling
                if args.ctm == 'True':
                    if args.tgtm == 'True':
                        sample = model.get_ctm_sample(Y.cuda(), args.N, model.dnn_tgt)
                    else:
                        sample = model.get_ctm_sample(Y.cuda(), args.N, model.dnn)                    
                elif model.sde.__class__.__name__ == 'OUVESDE':
                    if args.sampler_type == 'pc':
                        sampler = model.get_pc_sampler('reverse_diffusion', args.corrector, Y.cuda(), N=args.N, corrector_steps=args.corrector_steps, snr=args.snr)
                    elif args.sampler_type == 'ode':
                        sampler = model.get_ode_sampler(Y.cuda(), N=args.N, atol=1e-6, rtol=1e-3)
                    sample, _ = sampler()                    
                elif model.sde.__class__.__name__ == 'SBVESDE':                
                    sampler_type = 'ode' if args.sampler_type == 'pc' else args.sampler_type
                    sampler = model.get_sb_sampler(sde=model.sde, y=Y.cuda(), sampler_type=sampler_type, N=args.N)
                    sample, _ = sampler()
                else:
                    # Profile
                    wrapped_model = SgmseWrapper(model, args)
                    macs, params = profile(wrapped_model, inputs=(Y.cuda(),))
                    macs /= 1e9 ## -> GMACs
                    macs /= args.N ## -> /step
                    macs /= T_orig/48000 ## -> / 1sec-data
                    flops = macs * 2 ## -> GFLOPs
                    params /= 1e6 ## -> Mega parameters
                    print(f'MACs: {macs:.2f}[GMACs/1sec-data/1-step], FLOPs: {flops:.2f}[GFLOPs/1sec-data/1-step], params: {params:.2f}[MPARAMs]')

                # Backward transform in time domain
                x_hat = model.to_audio(sample.squeeze(), T_orig) # sample: [1, 1, 768, 1024]

                # Renormalize
                x_hat = x_hat * norm_factor

                # Write enhanced wav file
                makedirs(dirname(join(args.enhanced_dir, filename)), exist_ok=True)
                write(join(args.enhanced_dir, filename), x_hat.cpu().numpy(), sr)
