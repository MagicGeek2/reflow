import ml_collections

import reflow.configs.default as default_config

# ! 必须定义 get_config 函数


def get_config():
    config = default_config.get_config()

    # diffusers
    config.diffusers = diffusers = ml_collections.ConfigDict()
    diffusers.ckpt_path = 'checkpoints/AltDiffusion'
    diffusers.tokenizer = 'xlm_roberta_tokenizer'
    diffusers.text_encoder = 'xlm_roberta_text_model'
    diffusers.vae = 'autoencoder_kl'
    diffusers.score_model = 'unet_2d_condition_model'
    diffusers.load_score_model = True
    diffusers.gradient_checkpointing = True
    diffusers.use_xformers = False

    # ema
    config.ema = ema = ml_collections.ConfigDict()
    ema.decay = 0.999

    # training
    training = config.training
    training.ckpt_path = None
    training.reduce_mean = True
    training.randz0 = 'fix' # NOTE : random noise for same target if 'random' ; else 'fix'
    training.num_steps = 220000
    training.batch_size = 3
    training.gradient_accumulation_steps = 1
    training.mixed_precision='no'
    # # ! debug only
    # training.log_freq = 9
    # training.eval_freq = 9
    # training.snapshot_freq = 10
    
    # sampling
    sampling = config.sampling
    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian'
    sampling.init_noise_scale = 1.0
    sampling.randz0 = 'fix'
    sampling.use_ode_sampler = 'euler'
    sampling.sample_N = 100

    # reflow
    config.reflow = reflow = ml_collections.ConfigDict()
    reflow.reflow_t_schedule = 'uniform' # NOTE: t0, t1, uniform, or an integer k > 1
    reflow.reflow_loss = 'l2'  # NOTE: l2, lpips, lpips+l2

    # data
    data = config.data
    data.train_root = 'data/coco2014_reflow/train5M'
    data.eval_root = 'data/coco2014_reflow/val10k'
    data.dl_workers = 4
    # data.centered = True
    
    optim=config.optim
    optim.use_8bit_adam = False
    optim.lr_scheduler='linear'
    
    # config.device=None

    return config