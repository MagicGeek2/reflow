from diffusers import AltDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from typing import Callable, List, Optional, Union
from argparse import Namespace
import time
import random
import numpy as np
from tqdm.auto import tqdm
import json
import multiprocessing as mp
import os
from pathlib import Path

def setup_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def pipe_main(
    pipeline,
    prompt: Union[str, List[str]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
):
    """adapted from pipeline.__call__ , return latents instead of images
    """
    # 0. Default height and width to unet
    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipeline.check_inputs(prompt, height, width, callback_steps)

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = pipeline._execution_device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_embeddings = pipeline._encode_prompt(
        prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # 4. Prepare timesteps
    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipeline.unet.in_channels
    latents = pipeline.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        text_embeddings.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - \
        num_inference_steps * pipeline.scheduler.order
    # with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat(
            [latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipeline.scheduler.scale_model_input(
            latent_model_input, t)

        # predict the noise residual
        noise_pred = pipeline.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipeline.scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # # call the callback, if provided
        # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
        #     progress_bar.update()
        #     if callback is not None and i % callback_steps == 0:
        #         callback(i, t, latents)

    # # 8. Post-processing
    # image = pipeline.decode_latents(latents)

    # # 9. Run safety checker
    # image, has_nsfw_concept = pipeline.run_safety_checker(image, device, text_embeddings.dtype)

    # # 10. Convert to PIL
    # if output_type == "pil":
    #     image = pipeline.numpy_to_pil(image)

    # if not return_dict:
    #     return (image, has_nsfw_concept)

    # return AltDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    return latents


def get_all_caps(split, dataset='coco2014'):
    assert dataset in ['coco2014']
    json_path = f'data/{dataset}/annotations/captions_{split}2014.json'
    print(f'load json file...')
    json_info = json.load(open(json_path, 'r'))
    print(f'done!')
    all_caps = []
    print(f'merge all captions to a list...')
    for a in tqdm(json_info['annotations']):
        cap = a['caption']
        cap = cap.replace('\n', '').strip(' ')
        all_caps.append(cap)
    print(f'done!')
    return all_caps


def repeat_over_limit(seq, limit):
    repeat = limit // len(seq)
    if limit % len(seq) != 0:
        repeat = repeat+1
    extended_seq = []
    for i in range(repeat):
        extended_seq.extend(seq)
    return extended_seq


def devide_to_groups(seq: list, num_groups):
    size_per_group = len(seq) // num_groups
    if len(seq) % num_groups != 0:
        size_per_group = size_per_group + 1
    groups = []
    groups_se = []
    idx = 0
    while idx < len(seq):
        s, e = idx, idx+size_per_group
        if e >= len(seq):
            e = len(seq)
        groups.append(seq[s:e])
        groups_se.append((s, e))
        idx = idx+size_per_group
    return groups, groups_se


def produce(caps, idx_se, device, bs, infer_steps, image_dir, ):
    pipeline = AltDiffusionPipeline.from_pretrained(
        "checkpoints/AltDiffusion",
        torch_dtype=torch.float16,
        requires_safety_checker=False
    )
    pipeline = pipeline.to(device)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config)

    i_start, i_end = idx_se
    idx = 0
    time_start = time.time()
    while idx < len(caps):
        if idx+bs>=len(caps):
            bs=len(caps)-idx
        prompt = caps[idx:idx+bs]
        s, e = idx+i_start, idx+bs+i_start
        rnd_latents = torch.randn(bs, 4, 64, 64).half()
        latent_code = pipe_main(
            pipeline, prompt, latents=rnd_latents, num_inference_steps=infer_steps)
        imgs2save = torch.stack(
            [rnd_latents.float(), latent_code.cpu().float()], dim=0).transpose(0, 1).numpy()
        for i, img2save in enumerate(imgs2save, start=s):
            np.save(os.path.join(image_dir, f'{i}.npy'), img2save)
        idx += bs

        print(
            f'{device}: [{idx}/{len(caps)}] ; time elapased {time.time()-time_start:.3f}')


def save_config(args):
    config2save = {
        "model": 'BAAI/AltDiffusion',
        "dataset": 'coco2014',
        "scheduler": 'DPMSolverMultistepScheduler'
    }
    config2save = {**config2save, **vars(args)}
    save_path = str(Path(args.save_dir).parent / 'index.json')
    json.dump(config2save, open(save_path, 'w'))


def prepare_args():
    args = Namespace()
    args.infer_steps = 25
    args.seed = 2333
    args.save_dir = 'data/coco2014_reflow/val/content'
    args.split = 'val'
    args.devices = [1,2,3,4,5] # ! specify your gpu_ids
    args.total_nums = 10000
    args.bs=16

    return args


if __name__ == "__main__":
    mp.set_start_method('spawn')

    args = prepare_args()

    setup_seed(args.seed)
    save_config(args)

    total_nums = args.total_nums
    all_caps = get_all_caps(args.split)
    extended_caps = repeat_over_limit(all_caps, total_nums)
    random.shuffle(extended_caps)
    extended_caps = extended_caps[:total_nums]

    device_list = [f'cuda:{i}' for i in args.devices]
    num_workers = len(device_list)

    txt_path = os.path.join(args.save_dir, 'captions.txt')
    # 可以一次性写入所有的captions
    print(f'writing txt file...')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(extended_caps))
    print(f'done!')

    groups, groups_se = devide_to_groups(extended_caps, num_workers)
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=produce, args=(
            groups[i],
            groups_se[i],
            device_list[i],
            args.bs,
            args.infer_steps,
            os.path.join(args.save_dir, 'images')
        ))
        p.start()
        workers.append(p)
        print(f'{device_list[i]} started')

    for p in workers:
        p.join()