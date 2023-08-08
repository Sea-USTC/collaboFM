try:
    import torch
except ImportError:
    torch = None

import random
import numpy as np

from tqdm import tqdm

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def calculate_normalization(cfg, client_manager):
    train_x=[]
    for i in range(cfg.federate.client_num):
        ds = client_manager.clients[i].train_ds
        train_x+=[ds[_][0].unsqueeze(0) for _ in tqdm(range(len(ds)))]
    train_x = torch.vstack(train_x)
    print(train_x.shape)
    train_x = train_x.transpose(0,1).flatten(start_dim=1)
    print(train_x.mean(dim=-1))
    print(train_x.std(dim=-1))
