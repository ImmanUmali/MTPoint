import os, pickle, time
from datetime import datetime

import numpy as np
import torch

import sys

def path_to_cousin(cousinname, filename):
    return '/'.join(
        (
            *os.path.abspath(__file__).split("\\")[:-2],
            cousinname,
            filename
        )
    )


def now_to_string(omit_year=False, omit_ms=False):
    ct = datetime.now()
    if omit_year:
        if omit_ms:
            return f"{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}"
        else:
            return f"{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}_{ct.microsecond//10000:02d}"
    if omit_ms:
        return f"{ct.year%100:02d}{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}"
    return f"{ct.year%100:02d}{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}_{ct.microsecond//10000:02d}"


def list2str(s, sep=','):
    return sep.join(list(map(str, s)))


def pickle_save(file, data, try_multiple_save=100):
    if not file.endswith('.pkl'):
        file += '.pkl'
    if try_multiple_save <= 0: try_multiple_save = 1
    for _ in range(try_multiple_save):
        try:
            with open(file, "wb") as fp:
                pickle.dump(data, fp)
            return
        except:
            time.sleep(0.5)
            continue
    raise ValueError("Save failed. Check file directory.")

        


def pickle_load(file):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    return data

    

def fourier_encode(x, max_freq, num_bands = 4):
    """
    Fourier feature postiion encodings
    reference: https://github.com/lucidrains/perceiver-pytorch
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[((None,) * (len(x.shape) - 1) + (...,))]

    x = x * scales * np.pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


def get_auto_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)