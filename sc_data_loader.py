"""
Code for loading in single-cell datasets
"""

import os
import logging
import anndata as ad
import collections
import numpy as np
import functools

from typing import *

import torch
from torch.utils.data import Dataset

import utils

class sc_Dataset(Dataset):
    def __init__(self, sc_file):
        self.data_raw = ad.read_h5ad(sc_file)
      
    def __len__(self):
        return len(self.data_raw.obs)

    def __getitem__(self, idx):
        x = utils.ensure_arr(self.data_raw.X[idx]).flatten()
        x_p = torch.from_numpy(x).type(torch.FloatTensor)
        return x_p
    
    @functools.lru_cache(32)
    def __get_chrom_idx(self) -> Dict[str, np.ndarray]:
        
        """
        Helper func for figuring out which feature indexes are on each chromosome
        """
        chromosomes = sorted(
            list(set(self.data_raw.var["chrom"]))
        )  # Sort to guarantee consistent ordering
        chrom_to_idx = collections.OrderedDict()
        for chrom in chromosomes:
            chrom_to_idx[chrom] = np.where(self.data_raw.var["chrom"] == chrom)
        return chrom_to_idx
    
    def get_per_chrom_feature_count(self) -> List[int]:
        """
        Return the number of features from each chromosome
        If we were to split a catted feature vector, we would split
        into these sizes
        """
        chrom_to_idx = self.__get_chrom_idx()
        return [len(indices[0]) for _chrom, indices in chrom_to_idx.items()]
    
    def get_per_chrom_name(self) -> List[str]:
        """
        Return the number of features from each chromosome
        If we were to split a catted feature vector, we would split
        into these sizes
        """
        chrom_to_idx = self.__get_chrom_idx()
        return [ chrom_idx for chrom_idx, indices in chrom_to_idx.items()]


   
class PairedDataset(Dataset):
    """
    Combines two datasets into one, where input is now (x1, x2) and
    output is (y1, y2). A Paired dataset simply combines x and y
    by returning the x input and y input as a tuple, and the x output
    and y output as a tuple, and does not "cross" between the datasets
    """
    def __init__(self, dataset_x, dataset_y):
        assert isinstance(
            dataset_x, Dataset
        ), f"Bad type for dataset_x: {type(dataset_x)}"
        assert isinstance(
            dataset_y, Dataset
        ), f"Bad type for dataset_y: {type(dataset_y)}"
        assert len(dataset_x) == len(dataset_y), "Mismatched length"

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
    
    def __len__(self):
        return len(self.dataset_x)
        
    # Inherits the init from SplicedDataset since we're doing the same thing - recording
    # the two different datasets
    def __getitem__(self, i):
        x1 = self.dataset_x[i]
        x2 = self.dataset_y[i]

        x_pair = (x1, x2)
        y_pair = (x1, x2)
        retval = torch.cat(x_pair), torch.cat(y_pair)
        
        return retval
    
class TripletsDataset(Dataset):
    """
    Combines two datasets into one, where input is now (x1, x2) and
    output is (y1, y2). A Paired dataset simply combines x and y
    by returning the x input and y input as a tuple, and the x output
    and y output as a tuple, and does not "cross" between the datasets
    """
    def __init__(self, dataset_x, dataset_y, dataset_rp):
        assert isinstance(
            dataset_x, Dataset
        ), f"Bad type for dataset_x: {type(dataset_x)}"
        assert isinstance(
            dataset_y, Dataset
        ), f"Bad type for dataset_y: {type(dataset_y)}"
        assert isinstance(
            dataset_rp, Dataset
        ), f"Bad type for dataset_rp: {type(dataset_rp)}"
        assert len(dataset_x) == len(dataset_y), "Mismatched length"
        assert len(dataset_x) == len(dataset_rp), "Mismatched length"

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.dataset_rp = dataset_rp
    
    def __len__(self):
        return len(self.dataset_x)
        
    # Inherits the init from SplicedDataset since we're doing the same thing - recording
    # the two different datasets
    def __getitem__(self, i):
        x0 = self.dataset_rp[i]
        x1 = self.dataset_x[i]
        x2 = self.dataset_y[i]

        x_pair = (x1, x0)
        y_pair = (x1, x2)
        retval = torch.cat(x_pair), torch.cat(y_pair)
        
        return retval
    
# class CombinedDataset(Dataset):
#     """
#     Combines two datasets into one, where input is now (x1, x2) and
#     output is (y1, y2). A Paired dataset simply combines x and y
#     by returning the x input and y input as a tuple, and the x output
#     and y output as a tuple, and does not "cross" between the datasets
#     """
#     def __init__(self, dataset_rna, dataset_atac, dataset_rp, dataset_degs):
#         assert isinstance(
#             dataset_rna, Dataset
#         ), f"Bad type for dataset_rna: {type(dataset_rna)}"
#         assert isinstance(
#             dataset_atac, Dataset
#         ), f"Bad type for dataset_atac: {type(dataset_atac)}"
#         assert isinstance(
#             dataset_rp, Dataset
#         ), f"Bad type for dataset_rp: {type(dataset_rp)}"
#         assert isinstance(
#             dataset_degs, Dataset
#         ), f"Bad type for dataset_degs: {type(datadataset_degsset_rp)}"
#         assert len(dataset_rna) == len(dataset_atac), "Mismatched length"
#         assert len(dataset_rna) == len(dataset_rp), "Mismatched length"
#         assert len(dataset_rna) == len(dataset_degs), "Mismatched length"

#         self.dataset_rna = dataset_rna
#         self.dataset_atac = dataset_atac
#         self.dataset_rp = dataset_rp
#         self.dataset_degs = dataset_degs
    
#     def __len__(self):
#         return len(self.dataset_rna)
        
#     # Inherits the init from SplicedDataset since we're doing the same thing - recording
#     # the two different datasets
#     def __getitem__(self, i):
#         rp = self.dataset_rp[i]
#         rna = self.dataset_rna[i]
#         atac = self.dataset_atac[i]
#         degs = self.dataset_degs[i]

#         x_pair = (rna, degs, atac, rp)
#         y_pair = (rna, atac)
#         retval = torch.cat(x_pair), torch.cat(y_pair)
        
#         return retval

 

class CombinedDataset(Dataset):
    """
    Combines two datasets into one, where input is now (x1, x2) and
    output is (y1, y2). A Paired dataset simply combines x and y
    by returning the x input and y input as a tuple, and the x output
    and y output as a tuple, and does not "cross" between the datasets
    """
    def __init__(self, dataset_rna, dataset_atac, dataset_rp):
        assert isinstance(
            dataset_rna, Dataset
        ), f"Bad type for dataset_rna: {type(dataset_rna)}"
        assert isinstance(
            dataset_atac, Dataset
        ), f"Bad type for dataset_atac: {type(dataset_atac)}"
        assert isinstance(
            dataset_rp, Dataset
        ), f"Bad type for dataset_rp: {type(dataset_rp)}"
        assert len(dataset_rna) == len(dataset_atac), "Mismatched length"
        assert len(dataset_rna) == len(dataset_rp), "Mismatched length"

        self.dataset_rna = dataset_rna
        self.dataset_atac = dataset_atac
        self.dataset_rp = dataset_rp
    
    def __len__(self):
        return len(self.dataset_rna)
        
    # Inherits the init from SplicedDataset since we're doing the same thing - recording
    # the two different datasets
    def __getitem__(self, i):
        rp = self.dataset_rp[i]
        rna = self.dataset_rna[i]
        atac = self.dataset_atac[i]

        x_pair = (rna, atac, rp)
        y_pair = (rna, atac)
        retval = torch.cat(x_pair), torch.cat(y_pair)
        
        return retval


def LoadH5ad(datadir, sets):
    """
    """
    sc_atac_dataset = sc_Dataset(os.path.join(datadir, sets+"_atac.h5ad"))
    sc_rna_dataset = sc_Dataset(os.path.join(datadir, sets+"_rna.h5ad"))
    sc_rp_dataset = sc_Dataset(os.path.join(datadir, sets+"_rp.h5ad"))

    return sc_atac_dataset, sc_rna_dataset, sc_rp_dataset