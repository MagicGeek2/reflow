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
    training.num_steps = 20
    training.batch_size = 8
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
    reflow.reflow_t_schedule = 't0' # NOTE: t0, t1, uniform, or an integer k > 1
    reflow.reflow_loss = 'lpips'  # NOTE: l2, lpips, lpips+l2
    # reflow.reflow_type = 'train_reflow'  # NOTE: generate_data_from_z0, train_reflow
    # reflow.last_flow_ckpt = 'ckpt_path' # NOTE: the rectified flow model to fine-tune
    # reflow.data_root = 'data_path'  # NOTE: the folder to load the generated data

    # data
    data = config.data
    data.root_dir = 'data/coco2014_reflow'
    data.dl_workers = 1
    # data.centered = True

    # # model
    # model = config.model
    # model.name = 'ncsnpp'
    # model.scale_by_sigma = False
    # model.ema_rate = 0.9999
    # model.normalization = 'GroupNorm'
    # model.nonlinearity = 'swish'
    # model.nf = 128
    # model.ch_mult = (1, 2, 2, 2)
    # model.num_res_blocks = 4
    # model.attn_resolutions = (16,)
    # model.resamp_with_conv = True
    # model.conditional = True
    # model.fir = False
    # model.fir_kernel = [1, 3, 3, 1]
    # model.skip_rescale = True
    # model.resblock_type = 'biggan'
    # model.progressive = 'none'
    # model.progressive_input = 'none'
    # model.progressive_combine = 'sum'
    # model.attention_type = 'ddpm'
    # model.init_scale = 0.
    # model.embedding_type = 'positional'
    # model.fourier_scale = 16
    # model.conv_size = 3
    
    optim=config.optim
    optim.use_8bit_adam = True

    return config
