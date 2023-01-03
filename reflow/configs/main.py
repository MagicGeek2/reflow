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
    diffusers.load_score_model = False
    diffusers.gradient_checkpointing = True

    # ema
    config.ema = ema = ml_collections.ConfigDict()
    ema.decay = 0.999

    # training
    training = config.training
    training.ckpt_path = None
    training.snapshot_sampling = True
    training.sample_randz = False
    training.reduce_mean = True
    training.num_steps = 1_000_000
    training.batch_size = 16
    # training.sde = 'rectified_flow'
    # training.continuous = False

    # sampling
    sampling = config.sampling
    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian'
    sampling.init_noise_scale = 1.0
    sampling.use_ode_sampler = 'rk45'

    # reflow
    config.reflow = reflow = ml_collections.ConfigDict()
    reflow.reflow_t_schedule = 'uniform' # NOTE: t0, t1, uniform, or an integer k > 1
    reflow.reflow_loss = 'l2'  # NOTE: l2, lpips, lpips+l2
    # reflow.reflow_type = 'train_reflow'  # NOTE: generate_data_from_z0, train_reflow
    # reflow.last_flow_ckpt = 'ckpt_path' # NOTE: the rectified flow model to fine-tune
    # reflow.data_root = 'data_path'  # NOTE: the folder to load the generated data

    # data
    data = config.data
    data.root_dir = 'data/coco2014_reflow'
    data.dl_workers = 1
    # data.centered = True
    
    optim=config.optim
    optim.use_8bit_adam = True

    return config
