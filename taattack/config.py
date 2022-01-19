import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEVICES = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
if len(DEVICES) == 0:
    DEVICES = ['cpu'] * 2
