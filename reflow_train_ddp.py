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
from typing import Iterable

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, XLMRobertaTokenizer
from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
from diffusers.utils.import_utils import is_xformers_available
from accelerate import Accelerator
from diffusers.optimization import get_scheduler

from reflow.utils import restore_checkpoint, save_checkpoint, optimization_manager, get_step_fn, get_sampling_fn, set_seed, decode_latents, get_loss_fn
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
    
    # score_model.to(config.device)
    
    if config.diffusers.gradient_checkpointing:
        score_model.enable_gradient_checkpointing()
        
    if config.diffusers.use_xformers:
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

    
    set_seed(config.seed)

    # Create directories for experimental logs
    sample_dir = workdir/"samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = workdir/"tensorboard"
    tb_dir.mkdir(exist_ok=True)
    writer = tensorboard.SummaryWriter(str(tb_dir))
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
    )
    
    if accelerator.is_main_process:
        logger.add(str(workdir / 'exp.log'))
        logger.info(f'\n{config}')
    

    tokenizer, text_encoder, vae, score_model = create_models(config)
    
    weight_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

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
    
    lr_scheduler = get_scheduler(
        config.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optim.warmup * config.training.gradient_accumulation_steps,
        num_training_steps=config.training.num_steps * config.training.gradient_accumulation_steps,
    )
    
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
    
    score_model, optimizer, train_dl, lr_scheduler = accelerator.prepare(
        score_model, optimizer, train_dl, lr_scheduler
    )
    train_iter = cycle(train_dl)
    eval_iter = cycle(eval_dl)
    accelerator.register_for_checkpointing(lr_scheduler)
    
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.ema.decay)
    
    # state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = dict(model=score_model, ema=ema, step=0)
    ckpt_path = config.training.ckpt_path
    if ckpt_path is not None:
        state = restore_checkpoint(ckpt_path, state, accelerator.device)
        accelerator.load_state(f'{ckpt_path}.acc')
    initial_step = int(state['step'])
    checkpoint_dir = workdir/'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("tmp", config=vars(config))

    sde = RectifiedFlow(
        init_type=config.sampling.init_type,
        noise_scale=config.sampling.init_noise_scale,
        reflow_flag=True,
        reflow_t_schedule=config.reflow.reflow_t_schedule,
        reflow_loss=config.reflow.reflow_loss,
        use_ode_sampler=config.sampling.use_ode_sampler,
        codec=vae,
        device=accelerator.device,
    )

    # Building sampling functions
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                        config.data.image_size, config.data.image_size)
    sampling_fn = get_sampling_fn(
        config, sde, sampling_shape, eps=1e-3)  # 使用 euler 或 rk45

    # optimize_fn = optimization_manager(config)
    # reduce_mean = config.training.reduce_mean
    
    # train_step_fn = get_step_fn(sde, train=True, optimize_fn=optimize_fn,
    #                             reduce_mean=reduce_mean, )
    # eval_step_fn = get_step_fn(sde, train=False, optimize_fn=optimize_fn,
    #                            reduce_mean=reduce_mean,)
    
    reduce_mean = config.training.reduce_mean
    train_loss_fn = get_loss_fn(sde, train=True, reduce_mean=reduce_mean,)
    eval_loss_fn = get_loss_fn(sde, train=False, reduce_mean=reduce_mean,)

    num_train_steps = config.training.num_steps
    if accelerator.is_main_process:
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

    pbar=trange(initial_step, num_train_steps, desc='Steps', disable=not accelerator.is_local_main_process)
    for step in pbar:
        train_loss=0.0
        for _ in range(config.training.gradient_accumulation_steps):
            batch=next(train_iter)
        # for accumulation_step, batch in enumerate(train_iter, start=1):
            with accelerator.accumulate(score_model):
                loss = train_loss_fn(state, prepare_step_fn_input(batch))
                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                train_loss += avg_loss
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(score_model.parameters(), config.optim.grad_clip)
                    
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # if accumulation_step==config.training.gradient_accumulation_steps:
                #     break
        train_loss = train_loss / config.training.gradient_accumulation_steps
        
        if accelerator.sync_gradients:
            state['step'] += 1
            state['ema'].update(score_model.parameters())
            pbar.set_postfix({'train_loss':train_loss.item()})
        
        if accelerator.is_main_process:
            if step % config.training.log_freq == 0:
                logger.info(f'step {step} | training_loss {train_loss.item():.5f}')
                writer.add_scalar("training_loss", train_loss, step)

            if step % config.training.eval_freq == 0:
                eval_batch = to_device(next(eval_iter), accelerator.device)
                eval_loss = eval_loss_fn(state, prepare_step_fn_input(eval_batch))
                logger.info(f'step {step} | eval_loss {eval_loss.item():.5f}')
                writer.add_scalar("eval_loss", eval_loss, step)
                
            if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps-1:
                # Save the checkpoint.
                accelerator.save_state(str(checkpoint_dir / f'checkpoint_s{step}.pth.acc'))
                save_checkpoint(
                    str(checkpoint_dir / f'checkpoint_s{step}.pth'), state)
                
            if step != 0 and step % config.training.sampling_freq == 0 or step == num_train_steps-1:
                # Generate and save samples
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                eval_batch = to_device(next(eval_iter), accelerator.device)
                eval_step_fn_input = prepare_step_fn_input(eval_batch)
                z0 = eval_step_fn_input.pop('z0')
                z1 = eval_step_fn_input.pop('z1')
                sample, n = sampling_fn(
                    score_model,
                    z=None if config.sampling.randz0 else z0,
                    condition=eval_step_fn_input,
                )
                ema.restore(score_model.parameters())

                images = decode_latents(vae, sample)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(images, nrow, padding=2)
                save_image(image_grid, str(sample_dir / f'sample_s{step}.png'))
                
    pbar.close()


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
