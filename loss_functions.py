"""
Loss functions
"""

import os
import sys
import functools
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter



class Warmup(object):  # Doesn't have to be nn.Module because it's not learned
    """
    Warmup layer similar to
    Sonderby 2016 - Linear deterministic warm-up
    """

    def __init__(self, inc: float = 5e-3, t_max: float = 1.0):
        self.t = 0.0
        self.t_max = t_max
        self.inc = inc
        self.counter = 0  # Track number of times called next

    def __iter__(self):
        return self

    def __next__(self):
        retval = self.t
        t_next = self.t + self.inc
        self.t = min(t_next, self.t_max)
        self.counter += 1
        return retval

class NullWarmup(Warmup):
    """
    No warmup - but provides a consistent API
    """

    def __init__(self, delay: int = 0, t_max: float = 1.0):
        self.val = t_max

    def __next__(self):
        return self.val


class BCELoss(nn.BCELoss):
    """Custom BCE loss that can correctly ignore the encoded latent space output"""

    def forward(self, x, target):
        input = x[0]
        return F.binary_cross_entropy(
            input, target, weight=self.weight, reduction=self.reduction
        )



class NegativeBinomialLoss(nn.Module):
    """
    Negative binomial loss. Preds should be a tuple of (mean, dispersion)
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        eps: float = 1e-10,
        l1_lambda: float = 0.0,
        mean: bool = True,
    ):
        super(NegativeBinomialLoss, self).__init__()
        self.loss = negative_binom_loss(
            scale_factor=scale_factor,
            eps=eps,
            mean=mean,
            debug=True,
        )
        self.l1_lambda = l1_lambda

    def forward(self, preds, target):
        preds, theta = preds[:2]
        l = self.loss(
            preds=preds,
            theta=theta,
            truth=target,
        )
        encoded = preds[:-1]
        l += self.l1_lambda * torch.abs(encoded).sum()
        return l




class QuadLoss(nn.Module):
    """
    Paired loss, but for the spliced autoencoder with 4 outputs
    """

    def __init__(
        self,
        loss1=NegativeBinomialLoss,
        loss2=BCELoss,
        loss2_weight: float = 3.0,
        cross_weight: float = 1.0,
        cross_warmup_delay: int = 0,
        link_strength: float = 0.0,
        link_func: Callable = lambda x, y: (x - y).abs().mean(),
        link_warmup_delay: int = 0,
        record_history: bool = False,
    ):
        super(QuadLoss, self).__init__()
        self.loss1 = loss1()
        self.loss2 = loss2()
        self.loss2_weight = loss2_weight
        self.history = []  # Eventually contains list of tuples per call
        self.record_history = record_history

        self.warmup = NullWarmup(t_max=link_strength)
        self.cross_warmup = NullWarmup(t_max=cross_weight)

        self.link_strength = link_strength
        self.link_func = link_func

    def get_component_losses(self, preds, target):
        """
        Return the four losses that go into the overall loss, without scaling
        """
        preds11, preds12, preds21, preds22 = preds
        if not isinstance(target, (list, tuple)):
            # Try to unpack into the correct parts
            target = torch.split(
                target, [preds11[0].shape[-1], preds22[0].shape[-1]], dim=-1
            )
        target1, target2 = target  # Both are torch tensors

        loss11 = self.loss1(preds11, target1)
        loss21 = self.loss1(preds21, target1)
        loss12 = self.loss2(preds12, target2)
        loss22 = self.loss2(preds22, target2)

        return loss11, loss21, loss12, loss22

    def forward(self, preds, target):
        loss11, loss21, loss12, loss22 = self.get_component_losses(preds, target)
        if self.record_history:
            detensor = lambda x: x.detach().cpu().numpy().item()
            self.history.append([detensor(l) for l in (loss11, loss21, loss12, loss22)])

        loss = loss11 + self.loss2_weight * loss22
        loss += next(self.cross_warmup) * (loss21 + self.loss2_weight * loss12)

        if self.link_strength > 0:
            l = next(self.warmup)
            if l > 1e-6:  # If too small we disregard
                preds11, preds12, preds21, preds22 = preds
                encoded1 = preds11[-1]  # Could be preds12
                encoded2 = preds22[-1]  # Could be preds21
                d = self.link_func(encoded1, encoded2)
                loss += self.link_strength * d
        return loss


def negative_binom_loss(
    scale_factor: float = 1.0,
    eps: float = 1e-10,
    mean: bool = True,
    debug: bool = False,
    tb: SummaryWriter = None,
) -> Callable:
    """
    Return a function that calculates the binomial loss
    https://github.com/theislab/dca/blob/master/dca/loss.py

    combination of the Poisson distribution and a gamma distribution is a negative binomial distribution
    """

    def loss(preds, theta, truth, tb_step: int = None):
        """Calculates negative binomial loss as defined in the NB class in link above"""
        y_true = truth
        y_pred = preds * scale_factor

        if debug:  # Sanity check before loss calculation
            assert not torch.isnan(y_pred).any(), y_pred
            assert not torch.isinf(y_pred).any(), y_pred
            assert not (y_pred < 0).any()  # should be non-negative
            assert not (theta < 0).any()

        # Clip theta values
        theta = torch.clamp(theta, max=1e6)

        t1 = (
            torch.lgamma(theta + eps)
            + torch.lgamma(y_true + 1.0)
            - torch.lgamma(y_true + theta + eps)
        )
        t2 = (theta + y_true) * torch.log1p(y_pred / (theta + eps)) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps))
        )
        if debug:  # Sanity check after calculating loss
            assert not torch.isnan(t1).any(), t1
            assert not torch.isinf(t1).any(), (t1, torch.sum(torch.isinf(t1)))
            assert not torch.isnan(t2).any(), t2
            assert not torch.isinf(t2).any(), t2

        retval = t1 + t2
        if debug:
            assert not torch.isnan(retval).any(), retval
            assert not torch.isinf(retval).any(), retval

        if tb is not None and tb_step is not None:
            tb.add_histogram("nb/t1", t1, global_step=tb_step)
            tb.add_histogram("nb/t2", t2, global_step=tb_step)

        return torch.mean(retval) if mean else retval

    return loss

