import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.snapshot_freq = 20000
    training.sampling_freq = 5000
    training.log_freq = 50
    training.eval_freq = 50

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()

    # data
    config.data = data = ml_collections.ConfigDict()
    data.image_size = 64
    data.random_flip = False
    data.num_channels = 4

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
    config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    return config
