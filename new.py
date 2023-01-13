import torch
import math
import torch.nn.functional as F
import logging
from diffusers import AutoencoderKL, UNet2DConditionModel

from reflow_train import cycle

if __name__ == "__main__":
    path='logs/flow_AltInit/checkpoints/checkpoint_s200000.pth.acc/optimizer.bin'
    sd = torch.load(path)
    sd
    ...