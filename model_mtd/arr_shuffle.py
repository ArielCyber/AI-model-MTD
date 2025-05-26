from typing import Optional
import numpy as np
from itertools import product
import torch

def shuffle(arr:np.ndarray, seed:Optional[int]=None, inplace:bool=True):
    shape = arr.shape

    if not inplace:
        ret = arr.copy().ravel()
    else:
        ret = arr.ravel()

    rng = np.random.default_rng(seed)
    idxs_perm = rng.permutation(arr.size)

    ret = ret[idxs_perm]

    return ret.reshape(shape), idxs_perm

def recover(arr:np.ndarray, idxs_perm, inplace:bool=True):
    shape = arr.shape

    if not inplace:
        ret = arr.copy().ravel()
    else:
        ret = arr.ravel()

    # ret= np.empty_like(ret)

    ret = ret[np.argsort(idxs_perm)]

    return ret.reshape(shape)


def shuffle_torch(arr, seed:Optional[int]=None, inplace:bool=True):
    if not inplace:
        ret = arr.clone()
    else:
        ret = arr

    n_axis0 = arr.shape[0]
    if seed:
        gen = torch.Generator(device=arr.device).manual_seed(seed)
    else:
        gen = None
    idxs_perm = torch.randperm(n_axis0, generator=gen, device=arr.device)
    ret = ret[idxs_perm]
    return ret, idxs_perm

def recover_torch(arr, idxs_perm, inplace:bool=True):
    if not inplace:
        ret = arr.clone()
    else:
        ret = arr

    idxs_perm = torch.argsort(idxs_perm, dim=0)
    ret = ret[idxs_perm]
    return ret
