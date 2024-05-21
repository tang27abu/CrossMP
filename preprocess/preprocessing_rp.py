import os
import sys
import argparse
import logging

import scanpy as sc
import pandas as pd
import numpy as np
import scipy

import f_utils

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
assert os.path.isdir(DATA_DIR)

logging.basicConfig(level=logging.INFO)

def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--datadir", required=True, type=str, help="Data Directory"
    )
    parser.add_argument(
        "--organism", type=str, required=True, choices=['mouse','human'], help="Only Mouse and Human Supports",
    )
    return parser




def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()
  

    HG38_GTF = os.path.join(DATA_DIR,"Homo_sapiens.GRCh38.100.gtf.gz")
    assert os.path.isfile(HG38_GTF)

    MM38_GTF = os.path.join(DATA_DIR,"Mus_musculus.GRCm38.98.chr.gtf.gz")
    assert os.path.isfile(MM38_GTF)

    if args.organism == "mouse":
        gtf_file = MM38_GTF
    else:
        gtf_file = HG38_GTF
        
    # read, aggregate
    rp_aggregated = sc.read_h5ad(os.path.join(args.datadir, "hc_rp.h5ad"))

    # annotated chrom
    genes_reordered, chroms_reordered = f_utils.reorder_genes_by_chrom(
        rp_aggregated.var_names, gtf_file=gtf_file, return_chrom=True
    )
    rp_aggregated = rp_aggregated[:, genes_reordered]

    rp_aggregated_annot = f_utils.annotate_chroms(rp_aggregated, gtf_file)


    #autosomes_only
    autosomal_idx = [
        i
        for i, chrom in enumerate(rp_aggregated_annot.var["chrom"])
        if f_utils.is_numeric(chrom.strip("chr"))
    ]
    rp_aggregated_annot = rp_aggregated_annot[:, autosomal_idx]

    # Sort by the observation names so we can combine datasets
    sort_order_idx = np.argsort(rp_aggregated_annot.obs_names)
    rp_aggregated_annot = rp_aggregated_annot[sort_order_idx, :]

    rp_aggregated_annot = f_utils.annotate_basic_adata_metrics(rp_aggregated_annot)

    rp_aggregated_annot = f_utils.normalize_count_table(  # Normalizes in place
                rp_aggregated_annot,
                size_factors= True,
                normalize= True,
                log_trans= False,
            )
 
    if not isinstance(rp_aggregated_annot.X, scipy.sparse.csr_matrix):
        rp_aggregated_annot.X = scipy.sparse.csr_matrix(rp_aggregated_annot.X)

    size_norm_counts_m = f_utils.size_norm_counts(rp_aggregated_annot)

    rna_parsed = sc.read_h5ad(os.path.join(args.datadir, "hc_rna.h5ad"))
    rp_aggregated_annot  = size_norm_counts_m[rna_parsed.obs_names]

    data_split_to_idx_reload = np.load(os.path.join(args.datadir,'data_split_to_idx.npy'), allow_pickle=True)

    train_indices = data_split_to_idx_reload[()]['train']
    valid_indices = data_split_to_idx_reload[()]['valid']
    test_indices = data_split_to_idx_reload[()]['test']


    # Split the dataset using the indices from data_split_idx
    train_rp_set = rp_aggregated_annot[train_indices]
    valid_rp_set = rp_aggregated_annot[valid_indices]
    test_rp_set = rp_aggregated_annot[test_indices]

    with open(os.path.join(args.datadir, "rp_genes.txt"), "w") as sink:
        for gene in rp_aggregated_annot.var_names:
            sink.write(gene + "\n")

    rp_aggregated_annot.write_h5ad(
        os.path.join(args.datadir, "hc_rp_size_norm.h5ad")
    )
    
    train_rp_set.write_h5ad(
        os.path.join(args.datadir, "train_rp.h5ad")
    )
   
    valid_rp_set.write_h5ad(
        os.path.join(args.datadir, "valid_rp.h5ad")
    )
    
    test_rp_set.write_h5ad(
        os.path.join(args.datadir, "test_rp.h5ad")
    )

    logging.info(
        f"Created train data split with {train_rp_set.n_obs} examples"
    )
            
    logging.info(
        f"Created valid data split with {valid_rp_set.n_obs} examples"
    )
                
    logging.info(
        f"Created test data split with {test_rp_set.n_obs} examples"
    )


    # obs_names_1 = set(test_atac_set.obs_names)
    # obs_names_2 = set(test_rp_set.obs_names)

    # # Check if the observation names are identical and in the same order
    # if obs_names_1 == obs_names_2:
    #     print("Observations in adata1 and adata2 are the same and in the same order.")
    # else:
    #     print("Observations in adata1 and adata2 are not the same or not in the same order.")






if __name__ == "__main__":
    main()
