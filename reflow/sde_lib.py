import torch
from loguru import logger
from diffusers.models.vae import AutoencoderKLOutput, DecoderOutput


class DummyCodec(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return DecoderOutput(sample=x)

    def encode(self, x):
        return AutoencoderKLOutput(latent_dist=x)

    def decode(self, x):
        return DecoderOutput(sample=x)


class RectifiedFlow():
    def __init__(self, init_type='gaussian', noise_scale=1.0, reflow_flag=False, reflow_t_schedule='uniform', reflow_loss='l2', use_ode_sampler='rk45', sigma_var=0.0, ode_tol=1e-5, sample_N=None, codec=None, device='cpu'):
        
        self.init_type = init_type

        self.noise_scale = noise_scale
        self.use_ode_sampler = use_ode_sampler
        self.ode_tol = ode_tol
        self.sigma_t = lambda t: (1. - t) * sigma_var
        logger.info(f'Init. Distribution Variance: {self.noise_scale}')
        logger.info(f'SDE Sampler Variance: {sigma_var}')
        logger.info(f'ODE Tolerence: {self.ode_tol}')
        
        if use_ode_sampler in ['euler']:
            if sample_N is not None:
                self.sample_N = sample_N
                logger.info(f'Number of sampling steps: {self.sample_N}')

        self.reflow_flag = reflow_flag
        if self.reflow_flag:
            self.reflow_t_schedule = reflow_t_schedule
            self.reflow_loss = reflow_loss
            if 'lpips' in reflow_loss:
                import lpips
                self.lpips_model = lpips.LPIPS(net='vgg')
                self.lpips_model.to(device).requires_grad_(False)
                
                self.codec = codec
                if self.codec is None:
                    self.codec=DummyCodec().to(device) 

    @property
    def T(self):
        return 1

    @torch.no_grad()
    def ode(self, init_input, model, reverse=False):
        # run ODE solver for reflow. init_input can be \pi_0 or \pi_1
        from reflow.utils import to_flattened_numpy, from_flattened_numpy, get_model_fn
        from scipy import integrate
        rtol = 1e-5
        atol = 1e-5
        method = 'RK45'
        eps = 1e-3

        # Initial sample
        x = init_input.detach().clone()

        model_fn = get_model_fn(model, train=False)
        shape = init_input.shape
        device = init_input.device

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model_fn(x, vec_t*999)
            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        if reverse:
            solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
        else:
            solution = integrate.solve_ivp(ode_func, (eps, self.T), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
        x = torch.tensor(
            solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
        nfe = solution.nfev
        logger.info(f'NFE: {nfe}')

        return x

    def get_z0(self, shape_tensor, train=True):
        if self.init_type == 'gaussian':
            # standard gaussian #+ 0.5
            return torch.randn_like(shape_tensor)*self.noise_scale
        else:
            raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED")
