import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path



LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def parse_opt(know=False):
    parser = argparse.Argumentparser()
    #parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    #parser.add_argument('--rect', action='store_true', help='rectangular training')
    #parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    opt = parser.parse_known_args()[0] if know else parser.parse_args()
    return opt

def main(opt):
    print(opt)
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)