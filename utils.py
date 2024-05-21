"""
Utility functions

Some functions live here because otherwise managing their import
in other places would be overly difficult
"""
import os
import sys
import functools
import logging
from typing import *
import itertools
import collections
import gzip

import numpy as np
import pandas as pd
import scipy
import scanpy as sc
from anndata import AnnData

import torch

import intervaltree as itree
import sortedcontainers


def ensure_arr(x) -> np.ndarray:
    """Return x as a np.array"""
    if isinstance(x, np.matrix):
        return np.squeeze(np.asarray(x))
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
        return x.toarray()
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    else:
        raise TypeError(f"Unrecognized type: {type(x)}")


def get_device(i: int = None) -> str:
    """Returns the i-th GPU if GPU is available, else CPU"""
    if torch.cuda.is_available() and isinstance(i, int):
        devices = list(range(torch.cuda.device_count()))
        device_idx = devices[i]
        torch.cuda.set_device(device_idx)
        d = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(d)
    else:
        d = torch.device("cpu")
    return d

