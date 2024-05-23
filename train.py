import os
import sys
import logging
import argparse
import copy
import functools
import itertools
import time

import numpy as np
import pandas as pd
import scipy.spatial
import scanpy as sc
import anndata as ad

import matplotlib.pyplot as plt
from skorch.helper import predefined_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import skorch
import skorch.helper

import autoencoders
import activations
import loss_functions
import utils
import plot_utils
import sc_data_loader


torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False


logging.basicConfig(level=logging.INFO)


def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--datadir", "-d", required=True, type=str, help="Directory of input data"
    )
    parser.add_argument(
        "--outdir", "-o", required=True, type=str, help="Directory to output to"
    )
    parser.add_argument(
        "--hidden", type=int, default=16, help="Hidden dimensions"
    )
    parser.add_argument(
        "--lossweight", type=float, default=1.33, help="Relative loss weight"
    )
    parser.add_argument(
        "--lr", "-l", type=float, default=0.01,  help="Learning rate"
    )
    parser.add_argument(
        "--batchsize", "-b", type=int,  default=512, help="Batch size"
    )
    parser.add_argument(
        "--earlystop", type=int, default=25, help="Early stopping after N epochs"
    )
    parser.add_argument(
        "--seed", type=int,  default=182822, help="Random seed to use"
    )
    parser.add_argument(
        "--device", default=0, type=int, help="Device to train on"
    )
    return parser




def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")

    rna_ratio = 1
    atac_ratio = 1

    current_directory = os.getcwd()
    args.outdir = os.path.join(current_directory, args.outdir)
    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}_training_{timestr}.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Borrow parameters
  
    outdir_name = args.outdir
    if not os.path.isdir(outdir_name):
        assert not os.path.exists(outdir_name)
        os.makedirs(outdir_name)
    
    logging.info("Loading data")
    
    sc_atac_train_dataset, sc_rna_train_dataset, sc_rp_train_dataset = sc_data_loader.LoadH5ad(args.datadir, "train")
    sc_atac_valid_dataset, sc_rna_valid_dataset, sc_rp_valid_dataset = sc_data_loader.LoadH5ad(args.datadir, "valid")
    sc_atac_test_dataset, sc_rna_test_dataset, sc_rp_test_dataset = sc_data_loader.LoadH5ad(args.datadir, "test")

    sc_dual_train_dataset = sc_data_loader.CombinedDataset(
        sc_rna_train_dataset, sc_atac_train_dataset, sc_rp_train_dataset
    )
    sc_dual_valid_dataset = sc_data_loader.CombinedDataset(
        sc_rna_valid_dataset, sc_atac_valid_dataset, sc_rp_valid_dataset
    )
    sc_dual_test_dataset = sc_data_loader.CombinedDataset(
        sc_rna_test_dataset, sc_atac_test_dataset, sc_rp_test_dataset 
    )
 
    d = utils.get_device(args.device)

    # Instantiate and train model
    model_class = autoencoders.AssymSplicedAutoEncoder
    spliced_net = autoencoders.SplicedAutoEncoderSkorchNet(
        module=model_class,
        module__hidden_dim=args.hidden,  # Based on hyperparam tuning
        module__rna_dim=sc_rna_train_dataset.data_raw.shape[1],
        module__atac_dim=sc_atac_train_dataset.get_per_chrom_feature_count(),
        module__rp_dim=sc_rp_train_dataset.data_raw.shape[1],
        module__rna_ratio=rna_ratio,
        module__atac_ratio=atac_ratio,
        module__final_activations1=[
            activations.Exp(),
            activations.ClippedSoftplus(),
        ],
        module__final_activations2=nn.Sigmoid(),
        module__flat_mode=True,
        module__seed=args.seed,
        lr=args.lr,  
        criterion=loss_functions.QuadLoss,
        criterion__loss2=loss_functions.BCELoss,  
        criterion__loss2_weight=args.lossweight,  
        criterion__record_history=True,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        device=d,
        batch_size=args.batchsize,
        max_epochs=300,
        callbacks=[
            skorch.callbacks.EarlyStopping(patience=args.earlystop),
            skorch.callbacks.LRScheduler(
                policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                **{
                "mode": "min",
                "factor": 0.1,
                "patience": 15,
                "min_lr": 1.0e-06,
                }
            ),
            skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
            skorch.callbacks.Checkpoint(
                dirname=outdir_name, fn_prefix="net_", monitor="valid_loss_best",
            ),
        ],
        train_split=skorch.helper.predefined_split(sc_dual_valid_dataset),
        iterator_train__num_workers=8,
        iterator_valid__num_workers=8,
    )
    
    
    logging.info(f"Using device {d}")

    logging.info("Training...")

    spliced_net.fit(sc_dual_train_dataset, y=None)

    logging.info("Evaluating on test set")

    logging.info("Evaluating ATAC > RNA")
    sc_atac_rna_test_preds = spliced_net.translate_2_to_1(sc_dual_test_dataset)
    sc_atac_rna_test_preds_anndata = sc.AnnData(
        sc_atac_rna_test_preds,
        var=sc_rna_test_dataset.data_raw.var,
        obs=sc_rna_test_dataset.data_raw.obs,
    )
    sc_atac_rna_test_preds_anndata.write_h5ad(
        os.path.join(outdir_name, "atac_rna_test_preds.h5ad")
    )
    fig = plot_utils.plot_scatter_with_r(
        sc_rna_test_dataset.data_raw.X,
        sc_atac_rna_test_preds,
        one_to_one=True,
        logscale=True,
        density_heatmap=True,
        title="ATAC > RNA (test set)",
        fname=os.path.join(outdir_name, f"atac_rna_scatter_log.png"),
    )
    plt.close(fig)

    logging.info("Evaluating RNA > ATAC")
    sc_rna_atac_test_preds = spliced_net.translate_1_to_2(sc_dual_test_dataset)
    sc_rna_atac_test_preds_anndata = sc.AnnData(
        sc_rna_atac_test_preds,
        var=sc_atac_test_dataset.data_raw.var,
        obs=sc_atac_test_dataset.data_raw.obs,
    )
    sc_rna_atac_test_preds_anndata.write_h5ad(
        os.path.join(outdir_name, "rna_atac_test_preds.h5ad")
    )
    fig = plot_utils.plot_auroc(
        sc_atac_test_dataset.data_raw.X,
        sc_rna_atac_test_preds,
        title_prefix="RNA > ATAC",
        fname=os.path.join(outdir_name, f"rna_atac_auroc.png"),
    )
    plt.close(fig)
    

    del spliced_net

if __name__ == "__main__":
    main()
