import random
import numpy as np
import torch
from pathlib import Path
from reflow.sde_lib import RectifiedFlow
from loguru import logger


def decode_latents(vae, latents, float=True, cpu=True, permute=False) -> torch.Tensor:
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    if float:
        image = image.float()
    if cpu:
        image=image.cpu()
    if permute:
        image=image.permute(0,2,3,1)
    return image


def get_rectified_flow_sampler(sde, shape, inverse_scaler=None, device='cuda'):
    """
    Get rectified flow sampler

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    def euler_sampler(model, z=None, condition=None):
        """The probability flow ODE sampler with simple Euler discretization.

        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # * 50.
                z0 = sde.get_z0(torch.zeros(
                    shape, device=device), train=False).to(device)
                x = z0.detach().clone()

            else:
                x = z

            model_fn = get_model_fn(model, train=False)

            # Uniform
            dt = 1./sde.sample_N
            eps = 1e-3  # default: 1e-3
            for i in range(sde.sample_N):

                num_t = i / sde.sample_N * (sde.T - eps) + eps
                t = torch.ones(shape[0], device=device) * num_t
                t = (999*t).long()
                # pred = model_fn(x, t*999)  # Copy from models/utils.py
                # compatible with diffusers
                pred = model_fn(x, timestep=t, **condition).sample

                # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
                sigma_t = sde.sigma_t(num_t)
                pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (
                    0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*x.detach().clone())

                x = x.detach().clone() + pred_sigma * dt + sigma_t * \
                    np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)

            # x = inverse_scaler(x)
            nfe = sde.sample_N
            return x, nfe

    def rk45_sampler(model, z=None, condition=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          model: A velocity model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            rtol = atol = sde.ode_tol
            method = 'RK45'
            eps = 1e-3

            # Initial sample
            if z is None:
                z0 = sde.get_z0(torch.zeros(
                    shape, device=device), train=False).to(device)
                x = z0.detach().clone()
            else:
                x = z

            model_fn = get_model_fn(model, train=False)

            def ode_func(t, x, condition):
                x = from_flattened_numpy(x, shape).to(
                    device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                vec_t = (999*vec_t).long()
                # compatible with diffusers
                drift = model_fn(x, timestep=t, **condition).sample

                return to_flattened_numpy(drift)

            from scipy import integrate

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method, args=(condition,))
            nfe = solution.nfev
            x = torch.tensor(
                solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # x = inverse_scaler(x)

            return x, nfe

    logger.info(f'Type of Sampler: {sde.use_ode_sampler}')
    if sde.use_ode_sampler == 'rk45':
        return rk45_sampler
    elif sde.use_ode_sampler == 'euler':
        return euler_sampler
    else:
        assert False, 'Not Implemented!'


def get_sampling_fn(config, sde, shape, inverse_scaler=None, eps=1e-3):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    if sampler_name.lower() == 'rectified_flow':
        sampling_fn = get_rectified_flow_sampler(
            sde=sde, shape=shape, device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


def get_rectified_flow_loss_fn(sde, train, reduce_mean=True, eps=1e-3):
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * \
        torch.sum(*args, **kwargs)

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
          model: A score model.
          batch: A mini-batch of training data.

        Returns:
          loss: A scalar that represents the average loss value across the mini-batch.
        """
        z1 = batch.pop('z1')
        device = z1.device
        if sde.reflow_flag:
            z0 = batch.pop('z0')
            condition = batch
        else:
            z0 = sde.get_z0(z1).to(device)
        zshape = z0.shape
        bs = zshape[0]

        if sde.reflow_flag:
            # distill for t = 0 (k=1)  # ! 实际上 t=eps , 会有问题吗?
            if sde.reflow_t_schedule == 't0':
                t = torch.zeros(bs, device=device) * (sde.T - eps) + eps
            # # reverse distill for t=1 (fast embedding)
            # elif sde.reflow_t_schedule == 't1':
            #     t = torch.ones(bs, device=device) * (sde.T - eps) + eps
            elif sde.reflow_t_schedule == 'uniform':  # train new rectified flow with reflow
                t = torch.rand(bs, device=device) * (sde.T - eps) + eps
            elif type(sde.reflow_t_schedule) == int:  # k > 1 distillation
                t = torch.randint(0, sde.reflow_t_schedule, (bs, ), device=device) * (
                    sde.T - eps) / sde.reflow_t_schedule + eps
            else:
                raise (NotImplementedError('non-existing reflow t schedule'))
        else:
            # standard rectified flow loss
            t = torch.rand(bs, device=device) * (sde.T - eps) + eps

        # t_expand = t.view(-1, 1, 1, 1).repeat(1,batch.shape[1], batch.shape[2], batch.shape[3])
        t_expand = t.view(-1, 1, 1, 1)
        zt = t_expand * z1 + (1.-t_expand) * z0
        target = z1 - z0

        model_fn = get_model_fn(model, train=train)
        t = (999*t).long()
        # score = model_fn(perturbed_data, t*999)  # Copy from models/utils.py
        # compatible with diffusers
        score = model_fn(zt, timestep=t, **condition).sample

        if sde.reflow_flag:
            # we found LPIPS loss is the best for distillation when k=1; but good to have a try
            if sde.reflow_loss == 'l2':
                # train new rectified flow with reflow or distillation with L2 loss
                losses = torch.square(score - target)
            elif sde.reflow_loss == 'lpips':
                assert sde.reflow_t_schedule == 't0'
                losses = sde.lpips_model(
                    decode_latents(sde.codec, z0+score, float=False, cpu=False),
                    decode_latents(sde.codec, z1, float=False, cpu=False),
                    normalize=True,
                )
            elif sde.reflow_loss == 'lpips+l2':
                assert sde.reflow_t_schedule == 't0'
                lpips_losses = sde.lpips_model(
                    decode_latents(sde.codec, z0+score, float=False, cpu=False),
                    decode_latents(sde.codec, z1, float=False, cpu=False),
                    normalize=True,
                ).view(zshape[0], 1)
                l2_losses = torch.square(
                    score - target).view(zshape[0], -1).mean(dim=1, keepdim=True)
                losses = lpips_losses + l2_losses
            else:
                raise (NotImplementedError('non-existing reflow loss'))
        else:
            losses = torch.square(score - target)

        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      optimize_fn: An optimization function.
      reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
      continuous: `True` indicates that the model is defined to take continuous time steps.
      likelihood_weighting: If `True`, weight the mixture of score matching losses according to
            https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

    Returns:
      A one-step function for training or evaluation.
    """

    if isinstance(sde, RectifiedFlow):
        loss_fn = get_rectified_flow_loss_fn(
            sde, train, reduce_mean=reduce_mean)
    else:
        raise ValueError(
            f"Discrete training for {sde.__class__.__name__} is not recommended.")

    def step_fn(state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          state: A dictionary of training information, containing the score model, optimizer,
           EMA status, and number of optimization steps.
          batch: A mini-batch of training/evaluation data.

        Returns:
          loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss

    return step_fn


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(
        optimizer,
        params,
        step,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip
    ):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(*args, **kwargs):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
                for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if not train:
            model.eval()
            return model(*args, **kwargs)
        else:
            model.train()
            return model(*args, **kwargs)

    return model_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def restore_checkpoint(ckpt_path, state, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        logger.warning(
            f"No checkpoint found at {str(ckpt_path)}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(str(ckpt_path), map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_path, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_path)


def set_seed(seed: int,):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
            seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Partially based on: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
                `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) /
                        (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']
