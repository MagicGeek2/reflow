import ml_collections

import reflow.configs.default as default_config

# ! 必须定义 get_config 函数


def get_config():
    config = default_config.get_config()

    # sampling
    sampling = config.sampling
    sampling.randz0 = 'fix'
    sampling.ckpt_path = 'logs/2_reflow_AltInit/checkpoints/score_model_s200009.pth'

    sampling.decode_noise = False
    sampling.decode_latent = False
    sampling.return_traj = False

    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian'
    sampling.init_noise_scale = 1.0
    sampling.batch_size = 2
    sampling.num_samples = 2
    sampling.use_ode_sampler = 'euler'
    sampling.sample_N = 1  # NOTE: only working for euler sampler

    # diffusers
    config.diffusers = diffusers = ml_collections.ConfigDict()
    diffusers.ckpt_path = 'checkpoints/AltDiffusion'
    diffusers.tokenizer = 'xlm_roberta_tokenizer'
    diffusers.text_encoder = 'xlm_roberta_text_model'
    diffusers.vae = 'autoencoder_kl'
    diffusers.score_model = 'unet_2d_condition_model'
    diffusers.load_score_model = False
    diffusers.gradient_checkpointing = False
    diffusers.use_xformers = False

    # ema
    config.ema = ema = ml_collections.ConfigDict()
    ema.decay = 0.999

    # # reflow
    # config.reflow = reflow = ml_collections.ConfigDict()
    # reflow.reflow_t_schedule = 'uniform' # NOTE: t0, t1, uniform, or an integer k > 1
    # reflow.reflow_loss = 'l2'  # NOTE: l2, lpips, lpips+l2

    # data
    data = config.data
    data.eval_root = 'data/coco2014_reflow/val10k'
    data.dl_workers = 1

    config.device = 'cpu'

    return config
