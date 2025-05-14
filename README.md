# Schrödinger Bridge Consistency Trajectory Models for Speech Enhancement

<br>

# 👉 [**Demo (Audio samples)**](https://raw.githack.com/s0h1u2/anonymous/refs/heads/main/sbctm_demo.html)

<br>

## Description

This repository is the official PyTorch implementation of "Schrödinger Bridge Consistency Trajectory Models for Speech Enhancement"

- Paper (WASPAA 2025) **(Not submitted during the review period)**
- [Pretrained model](https://osf.io/download/f293c/?view_only=e406b105dd274657b7b33cea9dc764af) (trained on the VoiceBank-DEMAND dataset downsampled to 16 kHz)
- [Demo](https://raw.githack.com/s0h1u2/anonymous/refs/heads/main/sbctm_demo.html) (Audio samples)

Contact: anonymous@xxxx.xxx **(Anonymous during the review period)**

## Installation

- Create a new virtual environment with Python 3.11.
- Install the package dependencies via `pip install -r requirements.txt`.
    - Let pip resolve the dependencies for you. If you encounter any issues, please check `requirements_version.txt` for the exact versions we used.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--nolog` to `train.py`.
    - Your logs will be stored as local CSVLogger logs in `lightning_logs/`.

## Training

Training is performed using `train.py`, for example, with the following command:
```bash
python train.py --teacher_path <teacher_ckpt_path> --base_dir <traindata_dir_path> --batch_size <batch_size> --max_epochs <max_epoch_num>
```
Main arguments:
- teacher_path: Path to the checkpoint as the teacher model
- base_dir: Path to the directory containing subdirectories `train/` and `valid/`, with the same filenames present in both (`.wav` files)
- batch_size: Batch size for taining (integer value)
- max_epochs: The number of epochs

**Note:**
- In our paper, we additionally set the arguments for `--backbone ncsnpp-ctm_v2 --sde sbve --opt radam --ctm_lr 8e-5 --ctm_rec_w 1e-3 --ctm_psq_w 5e-4 --dsm_rec_w 1e-3 --dsm_psq_w 5e-4`, in addition to the ones mentioned above.
- If you encounter GPU memory issues, try using the `--nf <channel_num>` option with a value less than 128.

## Inference (Enhancement)

Inference is performed using `enhancement.py`, for example, with the following command:
```bash
python enhancement.py --test_dir <noisy_dir_path> --enhanced_dir <out_dir_path> --ckpt <pretrained_ckpt_path> --N <nfe>
```
Main arguments:
- test_dir: Path to the directory including noisy speech data
- enhanced_dir: Path to the directory for inference output
- ckpt: Path to the checkpoint to be used
- N: The number of reverse diffusion steps (integer value)

## Citation

We kindly ask you to cite our papers in your publication when using any of our research or code:
**Anonymous during the review period**
```bib
@inproceedings{
    sbctm2025,
    author={Anonymous},
    title={{Schr\"odinger Bridge Consistency Trajectory Models for Speech Enhancement}},
    year={2025},
    booktitle={WASPAA 2025}
}
```

## References

Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.
> https://github.com/sp-uhh/sgmse

> https://github.com/sony/soundctm

## License

This repository is primarily licensed under the MIT License.  
Some portions are derived from the work by Signal Processing (SP), Universität Hamburg.  
Some files or components are derived from projects released under the Apache License 2.0.  
See the `LICENSE` file for full details.
