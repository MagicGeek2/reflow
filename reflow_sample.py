from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image
from loguru import logger
from copy import deepcopy

from reflow.utils import get_sampling_fn, set_seed, decode_latents
from reflow.sde_lib import RectifiedFlow
from reflow.data.reflow_with_text import DataPairsWithText, get_reflow_dataset
from reflow.data.utils import get_image_transforms, LMDB_ndarray
from reflow.utils import create_models, cycle, to_device

from reflow.configs.sample import get_config # no need to import, just for convenience


def main(argv):

    config, eval_folder = FLAGS.config, FLAGS.eval_folder
    eval_folder = Path(eval_folder)

    logger.add(f'{eval_folder}/sample.log')
    logger.info(f'\n{config}')

    sample_dir = eval_folder/"samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    other_dirs = ["traj", "noise", "latent"]
    for odir in other_dirs:
        (eval_folder/odir).mkdir(exist_ok=True)

    caption_path = eval_folder / "caption.txt"

    set_seed(config.seed)

    tokenizer, text_encoder, vae, score_model = create_models(config)

    vae.to(config.device).eval()
    text_encoder.to(config.device).eval()
    score_model.to(config.device).eval().requires_grad_(False)

    eval_ds = get_reflow_dataset(
        data_root=config.data.eval_root,
        tokenizer=tokenizer,
        src_type='lmdb',
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=config.sampling.batch_size,
        shuffle=False,
        num_workers=config.data.dl_workers,
        drop_last=True, 
    )
    eval_iter = cycle(eval_dl)

    ckpt_path = config.sampling.ckpt_path
    if ckpt_path=='':
        logger.info(f'no resumed ckpt')
        if config.diffusers.load_score_model:
            logger.info(f'use pretrained diffusers score model')
        else:
            logger.info(f'totally random score model')
    else:
        logger.info(f'load ckpt from {ckpt_path}')
        score_model.load_state_dict(torch.load(ckpt_path, map_location=score_model.device), strict=False)

    sde = RectifiedFlow(
        init_type=config.sampling.init_type,
        noise_scale=config.sampling.init_noise_scale,
        reflow_flag=True,
        use_ode_sampler=config.sampling.use_ode_sampler,
        sample_N=config.sampling.sample_N,
    )

    # Building sampling functions
    sampling_shape = (config.sampling.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = get_sampling_fn(
        config, sde, sampling_shape, eps=1e-3)  # 使用 euler 或 rk45

    def prepare_step_fn_input(batch):
        z0 = batch.pop('noise')
        z1 = batch.pop('latent')
        encoder_hidden_states = text_encoder(**batch)[0]
        return {
            'z0': z0,
            'z1': z1,
            'encoder_hidden_states': encoder_hidden_states,
        }

    sample_cnt = 0
    sample_total = config.sampling.num_samples

    caption_file = open(str(caption_path), 'w')
    pbar=tqdm(total=sample_total, desc='Samples')
    with torch.no_grad():
        stop_sampling = False
        for batch in eval_iter:
            batch=to_device(batch, config.device)
            bs = batch['input_ids'].shape[0]
            if sample_cnt+bs >= sample_total:
                stop_sampling = True
                bs=sample_total-sample_cnt
                
            if config.sampling.decode_noise:
                # decode noise and save
                images = decode_latents(vae, batch['noise'])
                images=images[:bs]
                for i, image in enumerate(images, start=sample_cnt):
                    save_image(image, str(eval_folder / "noise" / f'noise_{i}.png'))

            if config.sampling.decode_latent:
                # decode latent and save
                images = decode_latents(vae, batch['latent'])
                images=images[:bs]
                for i, image in enumerate(images, start=sample_cnt):
                    save_image(image, str(eval_folder / "latent" / f'latent_{i}.png'))

            # decode and save sampling captions
            captions = tokenizer.batch_decode(batch['input_ids'])[:bs]
            captions = [s[4:s.find("<pad>")-4] for s in captions]
            caption_file.write('\n'.join(captions)+'\n')

            # sample, decode and save samples
            batch_copy = deepcopy(batch)
            eval_step_fn_input = prepare_step_fn_input(batch_copy)
            z0 = eval_step_fn_input.pop('z0')
            z1 = eval_step_fn_input.pop('z1')
            sample, *ret = sampling_fn(
                score_model,
                z = None if config.sampling.randz0 == 'random' else z0,
                condition = eval_step_fn_input,
                return_traj = config.sampling.return_traj
            )
            
            images = decode_latents(vae, sample)
            images=images[:bs]
            for i, image in enumerate(images, start=sample_cnt):
                save_image(image, str(sample_dir / f'sample_{i}.png'))

            if config.sampling.return_traj:
                traj = ret[1][:bs]
                for i, traj_i in enumerate(traj, start=sample_cnt):
                    traj_i = decode_latents(vae, traj_i)
                    traj_i = make_grid(traj_i, nrow=len(traj_i), padding=2)
                    save_image(traj_i, str(eval_folder / "traj" / f'traj_{i}.png'))

            sample_cnt += bs
            pbar.update(bs)
            if stop_sampling:
                break


    caption_file.close()
    pbar.close()


if __name__ == "__main__":
    FLAGS = flags.FLAGS

    config_flags.DEFINE_config_file(
        "config", None, "Training configuration.", lock_config=True)
    # flags.DEFINE_string("workdir", None, "Work directory.")
    # flags.DEFINE_enum("mode", None, ["train", "eval", "reflow"], "Running mode")
    flags.DEFINE_string("eval_folder", None,
                        "The folder name for storing evaluation results")
    # flags.mark_flags_as_required(["workdir", "config", "mode"])
    flags.mark_flags_as_required(["eval_folder", "config"])

    app.run(main)
