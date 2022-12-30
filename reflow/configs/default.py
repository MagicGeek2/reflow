import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.snapshot_freq = 20000
    training.log_freq = 50
    training.eval_freq = 50
    # training.n_iters = 1300001
    # # store additional checkpoints for preemption in cloud computing environments
    # training.snapshot_freq_for_preemption = 10000
    # # produce samples at each snapshot.
    # training.snapshot_sampling = True
    # training.likelihood_weighting = False
    # training.continuous = True
    # training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    # sampling.n_steps_each = 1
    # sampling.noise_removal = True
    # sampling.probability_flow = False
    # sampling.snr = 0.16

    # sampling.sigma_variance = 0.0  # NOTE: sigma variance for turning ODE to SDE
    # sampling.init_noise_scale = 1.0
    # sampling.use_ode_sampler = 'rk45'
    # sampling.ode_tol = 1e-5
    # sampling.sample_N = 1000

    # # evaluation
    # config.eval = evaluate = ml_collections.ConfigDict()
    # evaluate.begin_ckpt = 9
    # evaluate.end_ckpt = 26
    # evaluate.batch_size = 1024
    # evaluate.enable_sampling = False
    # evaluate.num_samples = 50000
    # evaluate.enable_loss = False
    # evaluate.enable_bpd = False
    # evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.image_size = 64
    data.random_flip = True
    data.num_channels = 4
    # data.dataset = 'CIFAR10'
    # data.centered = False
    # data.uniform_dequantization = False

    # # model
    # config.model = model = ml_collections.ConfigDict()
    # model.sigma_min = 0.01
    # model.sigma_max = 50
    # model.num_scales = 1000
    # model.beta_min = 0.1
    # model.beta_max = 20.
    # model.dropout = 0.1
    # model.embedding_type = 'fourier'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.lr = 1e-4
    optim.betas = (0.9, 0.999)
    optim.weight_decay = 0.
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.
    # optim.optimizer = 'Adam'

    config.seed = 2333
    config.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
