import argparse
import pytorch_lightning as pl
import os
import torch as th
import warnings
import random
import wandb
import numpy as np

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sgmse.util.other import set_torch_cuda_arch_list
import torch.distributed as dist
set_torch_cuda_arch_list()
th.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule

from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel
from sgmse.model import CTModel


def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # Set same seed on all GPUs
     pl.seed_everything(42)
     th.backends.cudnn.deterministic = True
     th.backends.cudnn.benchmark = False

     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = argparse.ArgumentParser(add_help=False)
     parser = argparse.ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp-ctm_48k")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--save_models_wb", action='store_true', help="Save models on WandB.")
          parser_.add_argument("--wandb_name", type=str, default="None", help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--wandb_runid", type=str, default="None", help="Run ID of W&B.")
          parser_.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
          parser_.add_argument("--ctm", type=str, default="True", help="Enable CTM")
          parser_.add_argument("--ckpt_dir", type=str, default="None", help="Checkpoint directory for save.")
          parser_.add_argument("--ckpt_path", type=str, default="None", help="Checkpoint path for load.")
          parser_.add_argument("--teacher_path", type=str, default=None, help="Path to teacher model")
     temp_args, _ = base_parser.parse_known_args()
     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     trainer_parser.add_argument("--sync_batchnorm", type=str, default="False", help="Synchronize batch normalization")
     trainer_parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
     trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs.")
     ScoreModel.add_argparse_args(parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     CTModel.add_argparse_args(parser.add_argument_group("CTModel", description=CTModel.__name__))
     sde_class.add_argparse_args(parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     if args.ckpt_path == "None" or not os.path.exists(args.ckpt_path):
          ckpt_path = None
     else:
          ckpt_path = args.ckpt_path
     
     if ckpt_path != None and os.path.exists(ckpt_path):
          init_with_teacher = False
     else:
          init_with_teacher = True
     
     if args.wandb_name == "None":
          wandb_name = None
     else:
          wandb_name = args.wandb_name

     if args.wandb_runid == "None":
          wandb_runid = None
     else:
          wandb_runid = args.wandb_runid

     if args.ckpt_dir != None and args.ckpt_dir != "None":
          os.makedirs(args.ckpt_dir, exist_ok=True)

     if args.ctm:
          # Load teacher model
          print(f'[INF]: Load teacher model')
          teacher = ScoreModel.load_from_checkpoint(args.teacher_path, base_dir='', strict=False)
          # Load student model
          print(f'[INF]: Create student model')
          model = CTModel(args.backbone, args.sde, data_module_cls=data_module_cls, teacher=teacher, ckpt_dir=args.ckpt_dir, **{**vars(arg_groups['CTModel']), **vars(arg_groups['SDE']), **vars(arg_groups['Backbone']), **vars(arg_groups['DataModule'])})
          if 0: # Display checkpoint keys
               checkpoint = th.load(ckpt_path)
               ckpt_keys = set(checkpoint['state_dict'].keys())
               model_keys = set(model.state_dict().keys())
               loaded_keys = ckpt_keys.intersection(model_keys)
               missing_keys = model_keys - ckpt_keys
               unexpected_keys = ckpt_keys - model_keys
               print('Loaded keys:', loaded_keys)
               print('Missing keys:', missing_keys)
               print('Unexpected keys:', unexpected_keys)
          if init_with_teacher:
               print(f'[INF]: Initialize student model with teacher model ({args.teacher_path})')
               src_params = dict(model.teacher.dnn.named_parameters())
               dst_params = dict(model.dnn.named_parameters())
               offset = -3
               for dst_name, dst in dst_params.items():
                    if dst_name in ["output_layer.weight", "output_layer.bias", "all_modules.0.W"]:
                         src = src_params[dst_name]
                         if dst.shape != src.shape:
                              min_numel = min(src.data.numel(), dst.data.numel())
                              dst.data.view(-1)[:min_numel].copy_(src.data.view(-1)[:min_numel])
                              #print(f'Size mismatch ! {dst_name}, partially copied')
                         else:
                              dst.data.copy_(src.data)
                              #print(f"Copied {dst_name}")
                    elif dst_name == "all_modules.2.weight":
                         src = src_params["all_modules.1.weight"]
                         if dst.shape != src.shape:
                              min_numel = min(src.data.numel(), dst.data.numel())
                              dst.data.view(-1)[:min_numel].copy_(src.data.view(-1)[:min_numel])
                              #print(f'Size mismatch ! {dst_name}, partially copied')
                         else:
                              dst.data.copy_(src.data)
                              #print(f"Copied {dst_name}")
                    elif dst_name == "all_modules.2.bias":
                         src = src_params["all_modules.1.bias"]
                         if dst.shape != src.shape:
                              min_numel = min(src.data.numel(), dst.data.numel())
                              dst.data.view(-1)[:min_numel].copy_(src.data.view(-1)[:min_numel])
                              #print(f'Size mismatch ! {dst_name}, partially copied')
                         else:
                              dst.data.copy_(src.data)
                              #print(f"Copied {dst_name}")
                    elif dst_name == "all_modules.3.weight":
                         src = src_params["all_modules.2.weight"]
                         if dst.shape != src.shape:
                              min_numel = min(src.data.numel(), dst.data.numel())
                              dst.data.view(-1)[:min_numel].copy_(src.data.view(-1)[:min_numel])
                              #print(f'Size mismatch ! {dst_name}, partially copied')
                         else:
                              dst.data.copy_(src.data)
                              #print(f"Copied {dst_name}")
                    elif dst_name == "all_modules.3.bias":
                         src = src_params["all_modules.2.bias"]
                         if dst.shape != src.shape:
                              min_numel = min(src.data.numel(), dst.data.numel())
                              dst.data.view(-1)[:min_numel].copy_(src.data.view(-1)[:min_numel])
                              #print(f'Size mismatch ! {dst_name}, partially copied')
                         else:
                              dst.data.copy_(src.data)
                              #print(f"Copied {dst_name}")
                    elif dst_name in ["all_modules.1.W", "all_modules.4.weight", "all_modules.4.bias", "all_modules.5.weight", "all_modules.5.bias"]:
                         #print(f"No matching parameter in teacher for {dst_name} (as expected)")
                         pass                         
                    elif dst_name.startswith("all_modules."):
                         try:
                              parts = dst_name.split(".")
                              module_index = int(parts[1])  # parts[]:"all_modules.4." -> module_index:"4"
                              src_index = module_index + offset
                              src_name = dst_name.replace(f"all_modules.{module_index}.", f"all_modules.{src_index}.")
                              if src_name in src_params:
                                   src = src_params[src_name]
                                   if dst.shape != src.shape:
                                        min_numel = min(src.data.numel(), dst.data.numel())
                                        dst.data.view(-1)[:min_numel].copy_(src.data.view(-1)[:min_numel])
                                        #print(f"Size mismatch ! {dst_name}, partially copied")
                                   else:
                                        dst.data.copy_(src.data)
                                        #print(f"Copied {src_name} to {dst_name}")
                              else:
                                   print(f"No matching parameter in teacher for {dst_name} (expected {src_name})")
                         except ValueError:
                              print(f"Invalid module index in parameter name: {dst_name}")
                    else:
                         print(f"Invalid module index in parameter name: {dst_name}")

     else:
          model = ScoreModel(backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls, **{**vars(arg_groups['ScoreModel']), **vars(arg_groups['SDE']), **vars(arg_groups['Backbone']), **vars(arg_groups['DataModule'])})

     # Set up logger configuration
     if args.save_models_wb:
          log_model = True
     else:
          log_model = None
     if args.nolog:
          logger = None
     else:
          logger = WandbLogger(project="sgmse", log_model=log_model, save_dir="logs", name=wandb_name, resume="allow", id=wandb_runid)
          logger.experiment.log_code(".")
          print(f'[INF]: Run ID: {logger.experiment.id}')

     # Set up callbacks for logger
     if logger != None:
          # save last.ckpt at every 100 step , and epoch end
          callbacks = [ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=0, enable_version_counter=False, every_n_train_steps=100, save_on_train_epoch_end=True)]
          if args.ctm:
               checkpoint_callback_psq1 = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=1, monitor="PESQ_1",  mode="max", filename='{epoch}_{PESQ_1:.2f}', enable_version_counter=False)
               checkpoint_callback_psqn = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=1, monitor="PESQ_8",  mode="max", filename='{epoch}_{PESQ_8:.2f}', enable_version_counter=False)
               checkpoint_callback_psq1_tgt = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=1, monitor="PESQ_TGT_1", mode="max", filename='{epoch}_{PESQ_TGT_1:.2f}', enable_version_counter=False)
               checkpoint_callback_psqn_tgt = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=1, monitor="PESQ_TGT_8", mode="max", filename='{epoch}_{PESQ_TGT_8:.2f}', enable_version_counter=False)
               callbacks += [checkpoint_callback_psq1, checkpoint_callback_psq1_tgt, checkpoint_callback_psqn, checkpoint_callback_psqn_tgt]
          else:
               checkpoint_callback_pesq  = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=1, monitor="pesq",   mode="max", filename='{pesq:.2f}',   enable_version_counter=False)
               checkpoint_callback_sisdr = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=1, monitor="si_sdr", mode="max", filename='{si_sdr:.2f}', enable_version_counter=False)
               callbacks += [checkpoint_callback_pesq, checkpoint_callback_sisdr]
     else:
          callbacks = None

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy="ddp",
          logger=logger,
          log_every_n_steps=1,
          num_sanity_val_steps=0,
          callbacks=callbacks
     )

     # Train model
     trainer.fit(model, ckpt_path=ckpt_path)

     wandb.finish()