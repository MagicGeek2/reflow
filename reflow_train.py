from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from pathlib import Path
from torch.utils import tensorboard
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange
from torchvision.utils import make_grid, save_image
import numpy as np
from loguru import logger

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, XLMRobertaTokenizer
from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
from diffusers.utils.import_utils import is_xformers_available

from reflow.utils import restore_checkpoint, save_checkpoint, optimization_manager, get_step_fn, get_sampling_fn, set_seed, decode_latents
from reflow.utils import ExponentialMovingAverage
from reflow.sde_lib import RectifiedFlow
from reflow.data.reflow_with_text import DataPairsWithText
from reflow.data.utils import get_image_transforms

def to_device(data, device):
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                data[k] = v.to(device)
            except:
                ...
    return data


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def create_models(config):
    """
    tokenizer, text_encoder, vae 固定不变，使用 hugface from_pretrained 的加载方式

    score_model 类型: nn.Module -> ModelMixin ->  UNet2DConditionModel , 可以兼容 reflow 的 load 和 save 方式

    新建 score_model 的时候提供两种选择:

        1. 完全新建，从 unet config 中构建模型，不加载权重

            config = UNet2DConditionModel.load_config(
                'checkpoints/AltDiffusion', subfolder='unet')
            unet = UNet2DConditionModel.from_config(config)

        2. 加载预训练文生图模型的 unet 权重 (sd 或 altDiff 的 unet 权重)

            unet = UNet2DConditionModel.from_pretrained(
                'checkpoints/AltDiffusion', subfolder='unet')
    """

    _MODELS = {
        'clip_text_model': CLIPTextModel,
        'clip_tokenizer': CLIPTokenizer,

        'xlm_roberta_text_model': RobertaSeriesModelWithTransformation,
        'xlm_roberta_tokenizer': XLMRobertaTokenizer,

        'autoencoder_kl': AutoencoderKL,
        'unet_2d_condition_model': UNet2DConditionModel,
    }

    def create_submodel(name, part, ckpt_path, load_weights=True):
        model_cls = _MODELS[name]
        if load_weights:
            submodel = model_cls.from_pretrained(ckpt_path, subfolder=part)
        else:
            submodel_config = model_cls.load_config(ckpt_path, subfolder=part)
            submodel = model_cls.from_config(submodel_config)
        return submodel

    # create submodels
    tokenizer = create_submodel(
        config.diffusers.tokenizer, 'tokenizer', config.diffusers.ckpt_path)
    text_encoder = create_submodel(
        config.diffusers.text_encoder, 'text_encoder', config.diffusers.ckpt_path)
    vae = create_submodel(config.diffusers.vae, 'vae',
                          config.diffusers.ckpt_path)
    score_model = create_submodel(config.diffusers.score_model, 'unet',
                                  config.diffusers.ckpt_path, load_weights=config.diffusers.load_score_model)

    # freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # send to correct device
    vae.to(config.device)
    text_encoder.to(config.device)
    score_model.to(config.device)
    # computing optimization
    if config.diffusers.gradient_checkpointing:
        score_model.enable_gradient_checkpointing()
        
    try:
        score_model.enable_xformers_memory_efficient_attention()
    except Exception as e:
        logger.warning(
            f"Could not enable memory efficient attention. Make sure xformers is installed correctly and a GPU is available: {e}"
        )

    return tokenizer, text_encoder, vae, score_model


def main(argv):

    config, workdir = FLAGS.config, FLAGS.workdir
    workdir = Path(workdir)

    logger.add(str(workdir / 'exp.log'))
    logger.info(f'\n{config}')

    set_seed(config.seed)

    # Create directories for experimental logs
    sample_dir = workdir/"samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = workdir/"tensorboard"
    tb_dir.mkdir(exist_ok=True)
    writer = tensorboard.SummaryWriter(str(tb_dir))

    tokenizer, text_encoder, vae, score_model = create_models(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema.decay)

    # Initialize the optimizer
    if config.optim.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        score_model.parameters(),
        lr=config.optim.lr,
        betas=config.optim.betas,
        weight_decay=config.optim.weight_decay,
        eps=config.optim.eps,
    )

    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    ckpt_path = config.training.ckpt_path
    if ckpt_path is not None:
        state = restore_checkpoint(ckpt_path, state, config.device)
    initial_step = int(state['step'])
    checkpoint_dir = workdir/'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    # 迭代 dataloader 的每个 batch 应得到 noise, latent, input_ids, attention_masks
    train_ds = DataPairsWithText(
        data_root=config.data.root_dir,
        phase='train',
        tokenizer=tokenizer,
        image_transforms=get_image_transforms(
            train=True, random_flip=config.data.random_flip),
    )
    eval_ds = DataPairsWithText(
        data_root=config.data.root_dir,
        phase='val',
        tokenizer=tokenizer,
        image_transforms=get_image_transforms(train=False, ),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.dl_workers,
        # collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.dl_workers,
        # collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    train_iter = cycle(train_dl)
    eval_iter = cycle(eval_dl)

    sde = RectifiedFlow(
        init_type=config.sampling.init_type,
        noise_scale=config.sampling.init_noise_scale,
        reflow_flag=True,
        reflow_t_schedule=config.reflow.reflow_t_schedule,
        reflow_loss=config.reflow.reflow_loss,
        use_ode_sampler=config.sampling.use_ode_sampler,
        codec=vae,
        device=config.device,
    )

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = get_sampling_fn(
            config, sde, sampling_shape, eps=1e-3)  # 使用 euler 或 rk45

    optimize_fn = optimization_manager(config)
    reduce_mean = config.training.reduce_mean
    train_step_fn = get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                reduce_mean=reduce_mean, )
    eval_step_fn = get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                               reduce_mean=reduce_mean,)

    num_train_steps = config.training.num_steps
    logger.info(f'REFLOW T SCHEDULE: {config.reflow.reflow_t_schedule}')
    logger.info(f'LOSS: {config.reflow.reflow_loss}')
    logger.info(f"Starting reflow training loop at step {initial_step}.")

    def prepare_step_fn_input(batch):
        z0 = batch.pop('noise')
        z1 = batch.pop('latent')
        encoder_hidden_states = text_encoder(**batch)[0]
        return {
            'z0': z0,
            'z1': z1,
            'encoder_hidden_states': encoder_hidden_states,
        }

    # for step in tqdm(range(initial_step, num_train_steps), desc='Steps', ):
    for step in trange(initial_step, num_train_steps, desc='Steps'):
        # main training logic
        batch = to_device(next(train_iter), config.device)
        # Execute one training step
        loss = train_step_fn(state, prepare_step_fn_input(batch))

        if step % config.training.log_freq == 0:
            logger.info(f'step {step} | training_loss {loss.item():.5f}')
            writer.add_scalar("training_loss", loss, step)

        if step % config.training.eval_freq == 0:
            eval_batch = to_device(next(eval_iter), config.device)
            eval_loss = eval_step_fn(state, prepare_step_fn_input(eval_batch))
            logger.info(f'step {step} | eval_loss {eval_loss.item():.5f}')
            writer.add_scalar("eval_loss", eval_loss, step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps-1:
            # Save the checkpoint.
            save_checkpoint(
                str(checkpoint_dir / f'checkpoint_s{step}.pth'), state)

        if step % config.training.sampling_freq == 0:
            # Generate and save samples
            ema.store(score_model.parameters())
            ema.copy_to(score_model.parameters())
            eval_batch = to_device(next(eval_iter), config.device)
            eval_step_fn_input = prepare_step_fn_input(eval_batch)
            z0 = eval_step_fn_input.pop('z0')
            z1 = eval_step_fn_input.pop('z1')
            sample, n = sampling_fn(
                score_model,
                z=None if config.training.sample_randz else z0,
                condition=eval_step_fn_input
            )
            ema.restore(score_model.parameters())

            images = decode_latents(vae, sample)
            nrow = int(np.sqrt(sample.shape[0]))
            image_grid = make_grid(images, nrow, padding=2)
            save_image(image_grid, str(sample_dir / f'sample_s{step}.png'))


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    config_flags.DEFINE_config_file(
        "config", None, "Training configuration.", lock_config=True)
    flags.DEFINE_string("workdir", None, "Work directory.")
    # flags.DEFINE_enum("mode", None, ["train", "eval", "reflow"], "Running mode")
    flags.DEFINE_string("eval_folder", "eval",
                        "The folder name for storing evaluation results")
    # flags.mark_flags_as_required(["workdir", "config", "mode"])
    flags.mark_flags_as_required(["workdir", "config"])

    app.run(main)
