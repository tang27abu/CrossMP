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
    input_group = parser.add_argument_group()
    input_group.add_argument(
        "--atac", type=str, nargs="*", help="ATAC Data Files to Process",
    )
    input_group.add_argument(
        "--rna", type=str, nargs="*", help="RNA Data Files to Process",
    )
    parser.add_argument(
        "--organism", type=str, required=True, choices=['mouse','human'], help="Only Mouse and Human Supports",
    )
    parser.add_argument(
        "--outdir", required=True, type=str, help="Output Directory"
    )
    return parser




def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    HG38_GTF = os.path.join(DATA_DIR,"Homo_sapiens.GRCh38.100.gtf.gz")
    assert os.path.isfile(HG38_GTF)

    MM38_GTF = os.path.join(DATA_DIR,"Mus_musculus.GRCm38.98.chr.gtf.gz")
    assert os.path.isfile(MM38_GTF)

    if args.organism == "mouse":
        gtf_file = MM38_GTF
    else:
        gtf_file = HG38_GTF
    # read, aggregate
    rna_aggregated = f_utils.aggregate_adata(args.rna, "Gene Expression", 0)

    # annotated chrom
    genes_reordered, chroms_reordered = f_utils.reorder_genes_by_chrom(
        rna_aggregated.var_names, gtf_file=gtf_file, return_chrom=True
    )
    rna_aggregated = rna_aggregated[:, genes_reordered]

    rna_aggregated_annot = f_utils.annotate_chroms(rna_aggregated, gtf_file)


    #autosomes_only
    autosomal_idx = [
        i
        for i, chrom in enumerate(rna_aggregated_annot.var["chrom"])
        if f_utils.is_numeric(chrom.strip("chr"))
    ]
    rna_aggregated_annot = rna_aggregated_annot[:, autosomal_idx]

    # Sort by the observation names so we can combine datasets
    sort_order_idx = np.argsort(rna_aggregated_annot.obs_names)
    rna_aggregated_annot = rna_aggregated_annot[sort_order_idx, :]

    rna_aggregated_annot = f_utils.annotate_basic_adata_metrics(rna_aggregated_annot)

    rna_aggregated_annot = f_utils.filter_adata_cells_and_genes(
                rna_aggregated_annot,
                filter_cell_min_counts= None,
                filter_cell_max_counts= None,
                filter_cell_min_genes= 200,
                filter_cell_max_genes= 7000,
                filter_gene_min_counts= None,
                filter_gene_max_counts= None,
                filter_gene_min_cells= None,
                filter_gene_max_cells= None,
            )

    rna_aggregated_annot = f_utils.normalize_count_table(  # Normalizes in place
                rna_aggregated_annot,
                size_factors= True,
                normalize= True,
                log_trans= True,
            )
 
    clip = 0.5 # for rna-seq
    if clip > 0:
        assert isinstance(clip, float) and 0.0 < clip < 50.0
        logging.info(f"Clipping to {clip} percentile")
        clip_low, clip_high = np.percentile(
            rna_aggregated_annot.X.flatten(), [clip, 100.0 - clip]
        )
        if clip_low == clip_high == 0:
            logging.warning("Skipping clipping, as clipping intervals are 0")
        else:
            assert (
                clip_low < clip_high
            ), f"Got discordant values for clipping ends: {clip_low} {clip_high}"
            rna_aggregated_annot.X = np.clip(rna_aggregated_annot.X, clip_low, clip_high)

    if not isinstance(rna_aggregated_annot.X, scipy.sparse.csr_matrix):
        rna_aggregated_annot.X = scipy.sparse.csr_matrix(rna_aggregated_annot.X)

    size_norm_counts_m = f_utils.size_norm_counts(rna_aggregated_annot)

    # Access the data matrix (.X) of the AnnData object
    data_matrix = size_norm_counts_m.X.todense()

    # Check if there are any negative values in the data matrix
    has_negative_values = np.any(data_matrix < 0)

    # read, if it's hg19, repool bin
    atac_aggregated = f_utils.aggregate_adata(args.atac, "Peaks", 0)

    atac_aggregated = f_utils.annotate_chroms(atac_aggregated, gtf_file)

    #autosomes_only
    autosomal_idx = [
        i
        for i, chrom in enumerate(atac_aggregated.var["chrom"])
        if f_utils.is_numeric(chrom.strip("chr"))
    ]
    atac_aggregated = atac_aggregated[:, autosomal_idx]

    # Sort by the observation names so we can combine datasets
    sort_order_idx = np.argsort(atac_aggregated.obs_names)
    atac_aggregated = atac_aggregated[sort_order_idx, :]

    #RP score, remove the binarization
    atac_aggregated.X[atac_aggregated.X.nonzero()] = 1  # .X here is a csr matrix

    atac_aggregated = f_utils.annotate_basic_adata_metrics(atac_aggregated)

    atac_aggregated = f_utils.filter_adata_cells_and_genes(
                atac_aggregated,
                filter_cell_min_counts= None,
                filter_cell_max_counts= None,
                filter_cell_min_genes= None,
                filter_cell_max_genes= None,
                filter_gene_min_counts= 5,
                filter_gene_max_counts= None,
                filter_gene_min_cells= 5,
                filter_gene_max_cells= 0.1,
            )
    
    logging.info("Splitting ATAC with predefined split")

    atac_parsed_annot_predefined = atac_aggregated[
                [
                    i
                    for i in size_norm_counts_m.obs.index
                    if i in atac_aggregated.obs.index
                ],
            ]
    

    data_split_idx = f_utils.split_train_valid_test(atac_parsed_annot_predefined)
    np.save(os.path.join(args.outdir,'data_split_to_idx.npy'), data_split_idx)

    # Split the dataset using the indices from data_split_idx
    train_atac_set = atac_parsed_annot_predefined[data_split_idx["train"]]
    valid_atac_set = atac_parsed_annot_predefined[data_split_idx["valid"]]
    test_atac_set = atac_parsed_annot_predefined[data_split_idx["test"]]

    train_rna_set = size_norm_counts_m[train_atac_set.obs_names]
    valid_rna_set = size_norm_counts_m[valid_atac_set.obs_names]
    test_rna_set = size_norm_counts_m[test_atac_set.obs_names]
    rna_parsed_annot  = size_norm_counts_m[atac_parsed_annot_predefined.obs_names]

    assert train_atac_set.shape[0] ==  train_rna_set.shape[0] 
    assert valid_atac_set.shape[0] ==  valid_rna_set.shape[0] 
    assert test_atac_set.shape[0] ==  test_rna_set.shape[0] 
    assert atac_parsed_annot_predefined.shape[0] ==  rna_parsed_annot.shape[0] 

    with open(os.path.join(args.outdir, "rna_genes.txt"), "w") as sink:
        for gene in rna_parsed_annot.var_names:
            sink.write(gene + "\n")
    with open(os.path.join(args.outdir, "atac_bins.txt"), "w") as sink:
        for atac_bin in atac_parsed_annot_predefined.var_names:
            sink.write(atac_bin + "\n")

    rna_parsed_annot.write_h5ad(
        os.path.join(args.outdir, "hc_rna.h5ad")
    )
    
    atac_parsed_annot_predefined.write_h5ad(
        os.path.join(args.outdir, "hc_atac.h5ad")
    )

    train_rna_set.write_h5ad(
        os.path.join(args.outdir, "train_rna.h5ad")
    )
    train_atac_set.write_h5ad(
        os.path.join(args.outdir, "train_atac.h5ad")
    )
    valid_rna_set.write_h5ad(
        os.path.join(args.outdir, "valid_rna.h5ad")
    )
    valid_atac_set.write_h5ad(
        os.path.join(args.outdir, "valid_atac.h5ad")
    )
    test_rna_set.write_h5ad(
        os.path.join(args.outdir, "test_rna.h5ad")
    )
    test_atac_set.write_h5ad(
        os.path.join(args.outdir, "test_atac.h5ad")
    )

    logging.info(
        f"Created train data split with {train_atac_set.n_obs} examples"
    )
            
    logging.info(
        f"Created valid data split with {valid_atac_set.n_obs} examples"
    )
                
    logging.info(
        f"Created test data split with {test_atac_set.n_obs} examples"
    )

if __name__ == "__main__":
    main()
