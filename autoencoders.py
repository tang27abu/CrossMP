import os
import sys
import logging
from typing import List, Tuple, Union, Callable
import functools

import numpy as np
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import skorch
import skorch.utils


import activations

torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    def __init__(self, num_inputs: int, num_units=32, activation=nn.PReLU):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_units = num_units

        self.encode1 = nn.Linear(self.num_inputs, 64)
        nn.init.xavier_uniform_(self.encode1.weight)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = activation()

        self.encode2 = nn.Linear(64, self.num_units)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(num_units)
        self.act2 = activation()

    def forward(self, x):
        x = self.act1(self.bn1(self.encode1(x)))
        x = self.act2(self.bn2(self.encode2(x)))
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_units: int = 32,
        intermediate_dim: int = 64,
        activation=nn.PReLU,
        final_activation=None,
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_units = num_units

        self.decode1 = nn.Linear(self.num_units, intermediate_dim)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(intermediate_dim)
        self.act1 = activation()

        self.decode21 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode21.weight)
        self.decode22 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode22.weight)
        self.decode23 = nn.Linear(intermediate_dim, self.num_outputs)
        nn.init.xavier_uniform_(self.decode23.weight)

        self.final_activations = nn.ModuleDict()
        if final_activation is not None:
            if isinstance(final_activation, list) or isinstance(
                final_activation, tuple
            ):
                assert len(final_activation) <= 3
                for i, act in enumerate(final_activation):
                    if act is None:
                        continue
                    self.final_activations[f"act{i+1}"] = act
            elif isinstance(final_activation, nn.Module):
                self.final_activations["act1"] = final_activation
            else:
                raise ValueError(
                    f"Unrecognized type for final_activation: {type(final_activation)}"
                )

    def forward(self, x, size_factors=None):
        """include size factor here because we may want to scale the output by that"""
        x = self.act1(self.bn1(self.decode1(x)))

        retval1 = self.decode21(x)  # This is invariably the counts
        if "act1" in self.final_activations.keys():
            retval1 = self.final_activations["act1"](retval1)
        if size_factors is not None:
            sf_scaled = size_factors.view(-1, 1).repeat(1, retval1.shape[1])
            retval1 = retval1 * sf_scaled  # Elementwise multiplication

        retval2 = self.decode22(x)
        if "act2" in self.final_activations.keys():
            retval2 = self.final_activations["act2"](retval2)

        retval3 = self.decode23(x)
        if "act3" in self.final_activations.keys():
            retval3 = self.final_activations["act3"](retval3)

        return retval1, retval2, retval3

class ChromEncoder(nn.Module):
    """
    Consumes multiple inputs (i.e. one feature vector for each chromosome)
    After processing everything to be the same dimensionality, concatenate
    to form a single latent dimension
    """

    def __init__(
        self, num_inputs: List[int], latent_dim: int = 32, activation=nn.PReLU
    ):
        super(ChromEncoder, self).__init__()
        self.num_inputs = num_inputs
        self.act = activation

        self.initial_modules = nn.ModuleList()
        for n in self.num_inputs:
            assert isinstance(n, int)
            layer1 = nn.Linear(n, 32)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(32)
            act1 = self.act()
            layer2 = nn.Linear(32, 16)
            nn.init.xavier_uniform_(layer2.weight)
            bn2 = nn.BatchNorm1d(16)
            act2 = self.act()
            self.initial_modules.append(
                nn.ModuleList([layer1, bn1, act1, layer2, bn2, act2])
            )

        self.encode2 = nn.Linear(16 * len(self.num_inputs), latent_dim)
        nn.init.xavier_uniform_(self.encode2.weight)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.act2 = self.act()

    def forward(self, x):
        assert len(x) == len(
            self.num_inputs
        ), f"Expected {len(self.num_inputs)} inputs but got {len(x)}"
        enc_chroms = []
        for init_mod, chrom_input in zip(self.initial_modules, x):
            for f in init_mod:
                chrom_input = f(chrom_input)
            enc_chroms.append(chrom_input)
        enc1 = torch.cat(
            enc_chroms, dim=1
        )  # Concatenate along the feature dim not batch dim
        enc2 = self.act2(self.bn2(self.encode2(enc1)))
        return enc2


class ChromDecoder(nn.Module):
    """
    Network that is per-chromosome aware, but does not does not output
    per-chromsome values, instead concatenating them into a single vector
    """

    def __init__(
        self,
        num_outputs: List[int],  # Per-chromosome list of output sizes
        latent_dim: int = 32,
        activation=nn.PReLU,
        final_activations=[activations.Exp(), activations.ClippedSoftplus()],
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.latent_dim = latent_dim

        self.decode1 = nn.Linear(self.latent_dim, len(self.num_outputs) * 16)
        nn.init.xavier_uniform_(self.decode1.weight)
        self.bn1 = nn.BatchNorm1d(len(self.num_outputs) * 16)
        self.act1 = activation()

        self.final_activations = nn.ModuleDict()
        if final_activations is not None:
            if isinstance(final_activations, list) or isinstance(
                final_activations, tuple
            ):
                assert len(final_activations) <= 3
                for i, act in enumerate(final_activations):
                    if act is None:
                        continue
                    self.final_activations[f"act{i+1}"] = act
            elif isinstance(final_activations, nn.Module):
                self.final_activations["act1"] = final_activations
            else:
                raise ValueError(
                    f"Unrecognized type for final_activation: {type(final_activation)}"
                )

        self.final_decoders = nn.ModuleList()  # List[List[Module]]
        for n in self.num_outputs:
            layer0 = nn.Linear(16, 32)
            nn.init.xavier_uniform_(layer0.weight)
            bn0 = nn.BatchNorm1d(32)
            act0 = activation()
            # l = [layer0, bn0, act0]
            # for _i in range(len(self.final_activations)):
            #     fc_layer = nn.Linear(32, n)
            #     nn.init.xavier_uniform_(fc_layer.weight)
            #     l.append(fc_layer)
            # self.final_decoders.append(nn.ModuleList(l))
            layer1 = nn.Linear(32, n)
            nn.init.xavier_uniform_(layer1.weight)
            layer2 = nn.Linear(32, n)
            nn.init.xavier_uniform_(layer2.weight)
            layer3 = nn.Linear(32, n)
            nn.init.xavier_uniform_(layer3.weight)
            self.final_decoders.append(
                nn.ModuleList([layer0, bn0, act0, layer1, layer2, layer3])
            )

    def forward(self, x):
        x = self.act1(self.bn1(self.decode1(x)))
        # This is the reverse operation of cat
        x_chunked = torch.chunk(x, chunks=len(self.num_outputs), dim=1)

        retval1, retval2, retval3 = [], [], []
        for chunk, processors in zip(x_chunked, self.final_decoders):
            # Each processor is a list of 3 different decoders
            # decode1, bn1, act1, *output_decoders = processors
            decode1, bn1, act1, decode21, decode22, decode23 = processors
            chunk = act1(bn1(decode1(chunk)))
            temp1 = decode21(chunk)
            temp2 = decode22(chunk)
            temp3 = decode23(chunk)

            if "act1" in self.final_activations.keys():
                # temp1 = output_decoders[0](chunk)
                temp1 = self.final_activations["act1"](temp1)
                # retval1.append(temp1)
            if "act2" in self.final_activations.keys():
                # temp2 = output_decoders[1](chunk)
                temp2 = self.final_activations["act2"](temp2)
                # retval2.append(temp2)
            if "act3" in self.final_activations.keys():
                # temp3 = output_decoders[2](chunk)
                temp3 = self.final_activations["act3"](temp3)
                # retval3.append(temp3)
            retval1.append(temp1)
            retval2.append(temp2)
            retval3.append(temp3)
        retval1 = torch.cat(retval1, dim=1)
        retval2 = torch.cat(retval2, dim=1)
        retval3 = torch.cat(retval3, dim=1)
        return retval1, retval2, retval3


class AssymSplicedAutoEncoder(nn.Module):
    """
    Assymmetric spliced autoencoder where branch 2 is a chrom AE
    """

    def __init__(
        self,
        rna_dim: int,
        atac_dim: List[int],
        rp_dim: int,
        rna_ratio: float,
        atac_ratio: float,
        hidden_dim: int = 16,
        final_activations1: list = [activations.Exp(), activations.ClippedSoftplus()],
        final_activations2=nn.Sigmoid(),
        flat_mode: bool = True,  # Controls if we have to re-split inputs
        seed: int = 182822,
    ):

        nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.flat_mode = flat_mode
        self.rna_dim = rna_dim
        self.atac_dim = atac_dim
        self.rp_dim = rp_dim

        self.rna_ratio = rna_ratio
        self.atac_ratio = atac_ratio
        
        self.num_outputs1 = (
            len(final_activations1)
            if isinstance(final_activations1, (list, set, tuple))
            else 1
        )
        self.num_outputs2 = (
            len(final_activations2)
            if isinstance(final_activations2, (list, set, tuple))
            else 1
        )

        self.encoder_rna = Encoder(num_inputs=rna_dim, num_units=hidden_dim)
        self.encoder_atac = ChromEncoder(num_inputs=atac_dim, latent_dim=hidden_dim)
        self.encoder_rp = Encoder(num_inputs=rp_dim, num_units=hidden_dim)

        self.decoder1 = Decoder(
            num_outputs=rna_dim,
            num_units=hidden_dim,
            final_activation=final_activations1,
        )
        self.decoder2 = ChromDecoder(
            num_outputs=atac_dim,
            latent_dim=hidden_dim,
            final_activations=final_activations2,
        )
        
    def _combine_output_and_encoded(self, decoded, encoded, num_outputs: int):
        """
        Combines the output and encoded in a single output
        """
        if num_outputs > 1:
            retval = *decoded, encoded
        else:
            if isinstance(decoded, tuple):
                decoded = decoded[0]
            retval = decoded, encoded
        assert isinstance(retval, (list, tuple))
        assert isinstance(
            retval[0], (torch.TensorType, torch.Tensor)
        ), f"Expected tensor but got {type(retval[0])}"
        return retval
    
    def split_catted_input(self, x):
        """Split the input into chunks that goes to each input to model"""
        rna, atac, rp = torch.split(x, [self.rna_dim, sum(self.atac_dim), self.rp_dim], dim=-1)
        return (rna, torch.split(atac, self.atac_dim, dim=-1), rp)


    def forward(self, x, size_factors=None, mode: Union[None, Tuple[int, int]] = None):
        if self.flat_mode:
            x = self.split_catted_input(x)
        assert isinstance(x, (tuple, list))
        assert len(x) == 3, "There should be two inputs to spliced autoencoder"
        encoded_rna = self.encoder_rna(x[0])
        encoded_atac = self.encoder_atac(x[1])
        encoded_rp = self.encoder_rp(x[2])

        encoded1 = encoded_rna
        encoded2 = encoded_atac + self.atac_ratio*encoded_rp

        decoded11 = self.decoder1(encoded1)
        retval11 = self._combine_output_and_encoded(
            decoded11, encoded1, self.num_outputs1
        )
        decoded12 = self.decoder2(encoded1)
        retval12 = self._combine_output_and_encoded(
            decoded12, encoded1, self.num_outputs2
        )
        decoded22 = self.decoder2(encoded2)
        retval22 = self._combine_output_and_encoded(
            decoded22, encoded2, self.num_outputs2
        )
        decoded21 = self.decoder1(encoded2)
        retval21 = self._combine_output_and_encoded(
            decoded21, encoded2, self.num_outputs1
        )

        if mode is None:
            return retval11, retval12, retval21, retval22
        retval_dict = {
            (1, 1): retval11,
            (1, 2): retval12,
            (2, 1): retval21,
            (2, 2): retval22,
        }
        if mode not in retval_dict:
            raise ValueError(f"Invalid mode code: {mode}")
        return retval_dict[mode]
    

class SplicedAutoEncoderSkorchNet(skorch.NeuralNet):
    """
    Skorch wrapper for the SplicedAutoEncoder above.
    Mostly here to take care of how we calculate loss
    """

    def predict_proba(self, x):
        """
        Subclassed so that calling predict produces a tuple of 4 outputs
        """
        y_probas1, y_probas2, y_probas3, y_probas4 = [], [], [], []
        for yp in self.forward_iter(x, training=False):
            assert isinstance(yp, tuple)
            yp1 = yp[0][0]
            yp2 = yp[1][0]
            yp3 = yp[2][0]
            yp4 = yp[3][0]
            y_probas1.append(skorch.utils.to_numpy(yp1))
            y_probas2.append(skorch.utils.to_numpy(yp2))
            y_probas3.append(skorch.utils.to_numpy(yp3))
            y_probas4.append(skorch.utils.to_numpy(yp4))
        y_proba1 = np.concatenate(y_probas1)
        y_proba2 = np.concatenate(y_probas2)
        y_proba3 = np.concatenate(y_probas3)
        y_proba4 = np.concatenate(y_probas4)
        # Order: 1to1, 1to2, 2to1, 2to2
        return y_proba1, y_proba2, y_proba3, y_proba4

    def get_encoded_layer(self, x):
        """Get the encoded representation as a TUPLE of two elements"""
        encoded1, encoded2 = [], []
        for out11, out12, out21, out22 in self.forward_iter(x, training=False):
            encoded1.append(out11[-1])
            encoded2.append(out22[-1])
        return np.concatenate(encoded1, axis=0), np.concatenate(encoded2, axis=0)

    def translate_1_to_1(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[0][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def translate_1_to_2(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[1][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def translate_2_to_1(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[2][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def translate_2_to_2(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[3][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def score(self, true, pred):
        return self.get_loss(pred, true)



