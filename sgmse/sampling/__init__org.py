# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def step_denoise(start_x_t, y, sde, score_fn, timesteps_list, to_zero=False):
    predictor_cls = PredictorRegistry.get_by_name('reverse_diffusion')
    corrector_cls = CorrectorRegistry.get_by_name('ald')
    predictor = predictor_cls(sde, score_fn, probability_flow=False)
    corrector = corrector_cls(sde, score_fn, snr=0.5, n_steps=1)

    with torch.no_grad():
        if 1:            
            if start_x_t == None:
                x_t = sde.prior_sampling(y.shape, y).to(y.device) # x_T = y + sigma(T)
            else:
                x_t = start_x_t
            x_mean = torch.zeros_like(x_t)

            for b in range(len(timesteps_list)):
                timesteps = timesteps_list[b]
                x_t_b1 = x_t[b].unsqueeze(0)
                x_mean_b1 = x_mean[b].unsqueeze(0)
                y_b1 = y[b].unsqueeze(0)
                for i in range(len(timesteps)):
                    start_t = timesteps[i]
                    if i != len(timesteps) - 1:
                        stepsize = start_t - timesteps[i + 1]
                    elif to_zero:
                        stepsize = timesteps[-1] # eps -> 0
                    else:
                        break
                    if stepsize > 0.0:
                        vec_t = torch.ones(1, device=y.device) * start_t
                        x_t_b1, x_mean_b1 = corrector.update_fn(x_t_b1, vec_t, y_b1)
                        x_t_b1, x_mean_b1 = predictor.update_fn(x_t_b1, vec_t, y_b1, stepsize)
                x_t[b] = x_t_b1.squeeze(0)
                x_mean[b] = x_mean_b1.squeeze(0)
        else:
            if start_x_t == None:
                x_t = sde.prior_sampling(y.shape, y).to(y.device) # x_T = y + sigma(T)
            else:
                x_t = start_x_t
            x_mean = None
            
            for b in range(len(timesteps_list)):
                timesteps = timesteps_list[b]
                for i in range(len(timesteps)):
                    start_t = timesteps[i]
                    if i != len(timesteps) - 1:
                        stepsize = start_t - timesteps[i + 1]
                    elif to_zero:
                        stepsize = timesteps[-1] # eps -> 0
                    else:
                        return x_t, x_mean                    
                    vec_t = torch.ones(y.shape[0], device=y.device) * start_t
                    x_t, x_mean = corrector.update_fn(x_t, vec_t, y)
                    x_t, x_mean = predictor.update_fn(x_t, vec_t, y, stepsize)              
        return x_t, x_mean


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, y,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            for i in range(sde.N):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1] # from eps to 0
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = corrector.update_fn(xt, vec_t, y)
                xt, xt_mean = predictor.update_fn(xt, vec_t, y, stepsize)
            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)
            return x_result, ns
    
    return pc_sampler


def get_ode_sampler(
    sde, score_fn, y, inverse_scaler=None,
    denoise=True, rtol=1e-5, atol=1e-5,
    method='RK45', eps=3e-2, device='cuda', **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        if 1:
            _, x = predictor.update_fn(x, vec_eps, y, sde.N)
        else:
            _, x = predictor.update_fn(x, vec_eps, y)
        return x

    def drift_fn(x, t, y):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, t, y)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If not represent, sample the latent code from the prior distibution of the SDE.
            x = sde.prior_sampling(y.shape, y).to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, vec_t, y)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler
