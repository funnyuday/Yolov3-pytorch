import re
import os
import glob
import torch
import random
import logging
import numpy as np
from pathlib import Path

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

def increment_path(path, sep='', is_exist=False):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if not path.exists() or is_exist:
        return path
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return Path(f"{path}{sep}{n}")  # update path

def gen_list(path, file_name):
    print(f"==> Generate datasets list from {path}...")
    file_list = []
    for filepath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[-1].lower() in file_name:
                file_list.append(os.path.join(filepath, filename))
    return file_list

