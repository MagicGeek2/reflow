import torch
import math
import torch.nn.functional as F
import logging
from diffusers import AutoencoderKL, UNet2DConditionModel

from reflow_train import cycle

if __name__ == "__main__":
    # timesteps=torch.LongTensor([0,1,2,3])
    # t_embd = get_timestep_embedding(timesteps, 8)

    # logging.info('hello this is my first message')

    # unet=UNet2DConditionModel.from_pretrained('checkpoints/AltDiffusion', subfolder='unet')
    # config=  UNet2DConditionModel.load_config('checkpoints/AltDiffusion', subfolder='unet')
    # unet=UNet2DConditionModel.from_config(config)
    # print(unet)
    
    l=[1,2,3]
    l=cycle(l)

    total=4
    for i, s in enumerate(l):
        if i==total:
            break
        print(s)
        
    for i, s in enumerate(l):
        if i==total:
            break
        print(s)
    ...
