import torch
from torchaudio import load
import torchaudio
from pesq import pesq
from .other import si_sdr, pad_spec
from sgmse import sampling
import torch.distributed as dist
from librosa import resample

# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1


def evaluate_model(model, num_eval_files):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Split the evaluation files among the GPUs
    eval_files_per_gpu = num_eval_files // world_size

    clean_files = model.data_module.valid_set.clean_files[:num_eval_files]
    noisy_files = model.data_module.valid_set.noisy_files[:num_eval_files]
    # Select the files for this GPU
    if rank == world_size - 1:
        clean_files = clean_files[rank*eval_files_per_gpu:]
        noisy_files = noisy_files[rank*eval_files_per_gpu:]
    else:   
        clean_files = clean_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]
        noisy_files = noisy_files[rank*eval_files_per_gpu:(rank+1)*eval_files_per_gpu]

    _si_sdr = 0; _pesq = 0; _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load the clean and noisy speech
        x, _ = load(clean_file)
        y, _ = load(noisy_file)

        T_orig = x.size(1)
        x = x.squeeze().numpy()

        # Normalize per utterance
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y, mode="reflection")
        # Reverse sampling
        sampler = model.get_pc_sampler('reverse_diffusion', 'ald', Y.cuda(), N=N, corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()

        # Resample
        x_16k = resample(x, orig_sr=48000, target_sr=16000).squeeze()
        x_hat_16k = resample(x_hat, orig_sr=48000, target_sr=16000).squeeze()

        # Calc. metrics
        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(16000, x_16k, x_hat_16k, 'wb') 
        _estoi += stoi(x_16k, x_hat_16k, 16000, extended=True)
        
    return _pesq / len(clean_files), _si_sdr / len(clean_files), _estoi / len(clean_files)

def evaluate_model_ctm(model, num_eval_files, steps1, steps2):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_eval_files = min(len(model.data_module.valid_set.clean_files), num_eval_files)    
    # Split the evaluation files among the GPUs
    eval_files_per_gpu = num_eval_files // world_size

    clean_files = model.data_module.valid_set.clean_files[: num_eval_files]
    noisy_files = model.data_module.valid_set.noisy_files[: num_eval_files]
    # Select the files for this GPU
    if rank == world_size - 1:
        clean_files = clean_files[rank * eval_files_per_gpu :]
        noisy_files = noisy_files[rank * eval_files_per_gpu :]
    else:   
        clean_files = clean_files[rank * eval_files_per_gpu : (rank + 1) * eval_files_per_gpu]
        noisy_files = noisy_files[rank * eval_files_per_gpu : (rank + 1) * eval_files_per_gpu]
    
    sdr_1 = 0; psq_1 = 0; estoi_1 = 0; sdr_n = 0; psq_n = 0; estoi_n = 0
    sdr_tgt_1 = 0; psq_tgt_1 = 0; estoi_tgt_1 = 0; sdr_tgt_n = 0; psq_tgt_n = 0; estoi_tgt_n = 0
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, sr_x = load(clean_file)
        y, sr_y = load(noisy_file)
        T_orig = x.size(1)

        x = x.squeeze().numpy()
        # Prepare DNN input
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y, mode="reflection")

        # Reverse sampling            
        sample_1 = model.get_ctm_sample(Y.clone(), steps1, model.dnn)
        sample_n = model.get_ctm_sample(Y.clone(), steps2, model.dnn)
        # Freq->Time
        x_hat_1     = model.to_audio(sample_1.squeeze(), T_orig)
        x_hat_n     = model.to_audio(sample_n.squeeze(), T_orig)
        x_hat_1     = x_hat_1 * norm_factor
        x_hat_n     = x_hat_n * norm_factor
        x_hat_1     = x_hat_1.squeeze().cpu().numpy()
        x_hat_n     = x_hat_n.squeeze().cpu().numpy()
        if sr_x == 16000 and sr_y == 16000:
            x_16k = x
            x_hat_1_16k = x_hat_1
            x_hat_n_16k = x_hat_n
        else:
            # Resample
            x_16k           = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()
            x_hat_1_16k     = resample(x_hat_1, orig_sr=sr_x, target_sr=16000).squeeze()
            x_hat_n_16k     = resample(x_hat_n, orig_sr=sr_x, target_sr=16000).squeeze()
        # Calc. metrics
        sdr_1       += si_sdr(x, x_hat_1)
        sdr_n       += si_sdr(x, x_hat_n)
        psq_1       += pesq(16000, x_16k + 1e-8, x_hat_1_16k + 1e-8, 'wb')
        psq_n       += pesq(16000, x_16k + 1e-8, x_hat_n_16k + 1e-8, 'wb')
        estoi_1     += stoi(x_16k, x_hat_1_16k, 16000, extended=True)
        estoi_n     += stoi(x_16k, x_hat_n_16k, 16000, extended=True)
        if 0:
            # for target model
            sample_tgt_1 = model.get_ctm_sample(Y.clone(), 1, model.dnn_tgt)
            sample_tgt_n = model.get_ctm_sample(Y.clone(), steps, model.dnn_tgt)
            x_hat_tgt_1 = model.to_audio(sample_tgt_1.squeeze(), T_orig)
            x_hat_tgt_n = model.to_audio(sample_tgt_n.squeeze(), T_orig)
            x_hat_tgt_1 = x_hat_tgt_1 * norm_factor
            x_hat_tgt_n = x_hat_tgt_n * norm_factor
            x_hat_tgt_1 = x_hat_tgt_1.squeeze().cpu().numpy()
            x_hat_tgt_n = x_hat_tgt_n.squeeze().cpu().numpy()
            if sr_x == 16000 and sr_y == 16000:
                x_hat_tgt_1_16k = x_hat_tgt_1
                x_hat_tgt_n_16k = x_hat_tgt_n
            else:
                x_hat_tgt_1_16k = resample(x_hat_tgt_1, orig_sr=sr_x, target_sr=16000).squeeze()
                x_hat_tgt_n_16k = resample(x_hat_tgt_n, orig_sr=sr_x, target_sr=16000).squeeze()
            sdr_tgt_1   += si_sdr(x, x_hat_tgt_1)
            sdr_tgt_n   += si_sdr(x, x_hat_tgt_n)
            psq_tgt_1   += pesq(16000, x_16k + 1e-8, x_hat_tgt_1_16k + 1e-8, 'wb')
            psq_tgt_n   += pesq(16000, x_16k + 1e-8, x_hat_tgt_n_16k + 1e-8, 'wb')
            estoi_tgt_1 += stoi(x_16k, x_hat_tgt_1_16k, 16000, extended=True)
            estoi_tgt_n += stoi(x_16k, x_hat_tgt_n_16k, 16000, extended=True)

    sdr_1       = sdr_1 / len(clean_files)
    sdr_n       = sdr_n / len(clean_files)
    psq_1       = psq_1 / len(clean_files)
    psq_n       = psq_n / len(clean_files)
    estoi_1     = estoi_1 / len(clean_files)
    estoi_n     = estoi_n / len(clean_files)
    if 0:
        sdr_tgt_1   = sdr_tgt_1 / len(clean_files)
        sdr_tgt_n   = sdr_tgt_n / len(clean_files)
        psq_tgt_1   = psq_tgt_1 / len(clean_files)
        psq_tgt_n   = psq_tgt_n / len(clean_files)
        estoi_tgt_1 = estoi_tgt_1 / len(clean_files)
        estoi_tgt_n = estoi_tgt_n / len(clean_files)

    return sdr_1, psq_1, estoi_1, sdr_n, psq_n, estoi_n, sdr_tgt_1, psq_tgt_1, estoi_tgt_1, sdr_tgt_n, psq_tgt_n, estoi_tgt_n

def evaluate_model_ctm_nstep(model, num_eval_files, nfe_list, dnn):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_eval_files = min(len(model.data_module.valid_set.clean_files), num_eval_files)    
    # Split the evaluation files among the GPUs
    eval_files_per_gpu = num_eval_files // world_size

    clean_files = model.data_module.valid_set.clean_files[: num_eval_files]
    noisy_files = model.data_module.valid_set.noisy_files[: num_eval_files]
    # Select the files for this GPU
    if rank == world_size - 1:
        clean_files = clean_files[rank * eval_files_per_gpu :]
        noisy_files = noisy_files[rank * eval_files_per_gpu :]
    else:   
        clean_files = clean_files[rank * eval_files_per_gpu : (rank + 1) * eval_files_per_gpu]
        noisy_files = noisy_files[rank * eval_files_per_gpu : (rank + 1) * eval_files_per_gpu]
    
    sdr = [0] * len(nfe_list)
    psq = [0] * len(nfe_list)
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, sr_x = load(clean_file)
        y, sr_y = load(noisy_file)
        T_orig = x.size(1)

        x = x.squeeze().numpy()
        if sr_x == 16000:
            x_16k = x
        else:
            x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze()

        # Prepare DNN input
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y, mode="reflection")

        for n in range(len(nfe_list)):
            # Reverse sampling
            sample = model.get_ctm_sample(Y.clone(), nfe_list[n], dnn)
            # Freq->Time
            x_hat = model.to_audio(sample.squeeze(), T_orig)
            x_hat = x_hat * norm_factor
            x_hat = x_hat.squeeze().cpu().numpy()
            if sr_x == 16000:
                x_hat_16k = x_hat
            else:
                x_hat_16k = resample(x_hat, orig_sr=sr_x, target_sr=16000).squeeze()
            # Calc. metrics
            sdr[n] += si_sdr(x, x_hat)
            psq[n] += pesq(16000, x_16k, x_hat_16k, 'wb')

    for n in range(len(nfe_list)):
        sdr[n] = sdr[n] / len(clean_files)
        psq[n] = psq[n] / len(clean_files)

    return sdr, psq
