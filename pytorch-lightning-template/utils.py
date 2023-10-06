import torch
import numpy as np
import random
import os
import platform
from omegaconf import OmegaConf
import torch.distributed as dist


def init_exp(config):
    exp_dir = os.path.join('experiment', config.exp_name)
    os.makedirs(exp_dir)
    OmegaConf.save(config, os.path.join(exp_dir, 'config.yaml'))

    return exp_dir
