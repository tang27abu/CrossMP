import os
import sys
import shutil
import subprocess
import shlex
import logging
import glob
import platform

import intervaltree
import collections
from typing import *
import sortedcontainers
import gzip
import functools

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import scipy
import random
from anndata import AnnData



DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
assert os.path.isdir(DATA_DIR)


def get_ad_reader(fname, ft_type):
    retval = sc.read_h5ad(fname)
    assert retval.var['feature_types'][0] == ft_type
    logging.info(f"Read in {fname} for {retval.shape} (obs x var)")
    return retval


def _harmonize_atac_intervals(
    intervals_1: List[str], intervals_2: List[str]
) -> List[str]:
    """
    Given two files describing intervals, harmonize them by merging overlapping
    intervals for each chromosome
    """

    def interval_list_to_itree(
        l: List[str],
    ) -> Dict[str, intervaltree.IntervalTree]:
        """convert the dataframe to a intervaltree"""
        retval = collections.defaultdict(intervaltree.IntervalTree)
        for s in l:
            chrom, span = s.split(":")
            start, stop = map(int, span.split("-"))
            retval[chrom][start:stop] = None
        return retval

    itree1 = interval_list_to_itree(intervals_1)
    itree2 = interval_list_to_itree(intervals_2)

    # Merge the two inputs
    merged_itree = {}
    for chrom in itree1.keys():
        if chrom not in itree2:  # Unique to itree1
            merged_itree[chrom] = itree1[chrom]
        combined = itree1[chrom] | itree2[chrom]
        combined.merge_overlaps()
        merged_itree[chrom] = combined
    for chrom in itree2.keys():  # Unique to itree2
        if chrom not in merged_itree:
            merged_itree[chrom] = itree2[chrom]

    retval = []
    interval_spans = []
    for chrom, intervals in merged_itree.items():
        for i in sorted(intervals):
            interval_spans.append(i.end - i.begin)
            i_str = f"{chrom}:{i.begin}-{i.end}"
            retval.append(i_str)

    logging.info(
        f"Average/SD interval size after merging: {np.mean(interval_spans):.4f} {np.std(interval_spans):.4f}"
    )
    return retval


def harmonize_atac_intervals(*intervals: List[str]) -> List[str]:
    """
    Given multiple intervals, harmonize them
    >>> harmonize_atac_intervals(["chr1:100-200"], ["chr1:150-250"])
    ['chr1:100-250']
    >>> harmonize_atac_intervals(["chr1:100-200"], ["chr1:150-250"], ["chr1:300-350", "chr2:100-1000"])
    ['chr1:100-250', 'chr1:300-350', 'chr2:100-1000']
    """
    assert len(intervals) > 0
    if len(intervals) == 1:
        return intervals
    retval = _harmonize_atac_intervals(intervals[0], intervals[1])
    for i in intervals[2:]:
        retval = _harmonize_atac_intervals(retval, i)
    return retval

def interval_string_to_tuple(x: str) -> Tuple[str, int, int]:
    """
    Convert the string to tuple
    >>> interval_string_to_tuple("chr1:100-200")
    ('chr1', 100, 200)
    >>> interval_string_to_tuple("chr1:1e+06-1000199")
    ('chr1', 1000000, 1000199)
    """
    tokens = x.split(":")
    assert len(tokens) == 2, f"Malformed interval string: {x}"
    chrom, interval = tokens
    # if not chrom.startswith("chr"):
    #     logging.warn(f"Got noncanonical chromsome in {x}")
    start, stop = map(float, interval.split("-"))
    assert start < stop, f"Got invalid interval span: {x}"
    return (chrom, int(start), int(stop))

def interval_strings_to_itree(
    interval_strings: List[str],
) -> Dict[str, intervaltree.IntervalTree]:
    """
    Given a list of interval strings, return an itree per chromosome
    The data field is the index of the interval in the original list
    """
    interval_tuples = [interval_string_to_tuple(x) for x in interval_strings]
    retval = collections.defaultdict(intervaltree.IntervalTree)
    for i, (chrom, start, stop) in enumerate(interval_tuples):
        retval[chrom][start:stop] = i
    return retval



def get_indices_to_form_target_intervals(
    genomic_intervals: List[str], target_intervals: List[str]
) -> List[List[int]]:
    """
    Given a list of genomic intervals in string format, and a target set of similar intervals
    Return a list of indices to combine to map into the target
    """
    source_itree = interval_strings_to_itree(genomic_intervals)
    target_intervals = [interval_string_to_tuple(x) for x in target_intervals]

    retval = []
    for chrom, start, stop in target_intervals:
        overlaps = source_itree[chrom].overlap(start, stop)
        retval.append([o.data for o in overlaps])
    return retval

def combine_array_cols_by_idx(
    arr, idx: List[List[int]], combine_func: Callable = np.sum
) -> scipy.sparse.csr_matrix:
    """Given an array and indices, combine the specified columns, returning as a csr matrix"""
    if isinstance(arr, np.ndarray):
        arr = scipy.sparse.csc_matrix(arr)
    elif isinstance(arr, pd.DataFrame):
        arr = scipy.sparse.csc_matrix(arr.to_numpy(copy=True))
    elif isinstance(arr, scipy.sparse.csr_matrix):
        arr = arr.tocsc()
    elif isinstance(arr, scipy.sparse.csc_matrix):
        pass
    else:
        raise TypeError(f"Cannot combine array cols for type {type(arr)}")

    new_cols = []
    for indices in idx:
        if not indices:
            next_col = scipy.sparse.csc_matrix(np.zeros((arr.shape[0], 1)))
        elif len(indices) == 1:
            next_col = scipy.sparse.csc_matrix(arr.getcol(indices[0]))
        else:  # Multiple indices to combine
            col_set = np.hstack([arr.getcol(i).toarray() for i in indices])
            next_col = scipy.sparse.csc_matrix(
                combine_func(col_set, axis=1, keepdims=True)
            )
        new_cols.append(next_col)
    new_mat_sparse = scipy.sparse.hstack(new_cols).tocsr()
    assert (
        len(new_mat_sparse.shape) == 2
    ), f"Returned matrix is expected to be 2 dimensional, but got shape {new_mat_sparse.shape}"
    # print(arr.shape, new_mat_sparse.shape)
    return new_mat_sparse


def repool_atac_bins(x: AnnData, target_bins: List[str]) -> AnnData:
    """
    Re-pool data from x to match the given target bins, summing overlapping entries
    """
    # TODO compare against __pool_features and de-duplicate code
    idx = get_indices_to_form_target_intervals(
        x.var.index, target_intervals=target_bins
    )
    # This already gives a sparse matrix
    data_raw_aggregated = combine_array_cols_by_idx(x.X, idx)
    retval = AnnData(
        data_raw_aggregated,
        obs=x.obs,
        var=pd.DataFrame(index=target_bins),
    )
    return retval




def liftover_intervals(
    intervals: List[str],
    chain_file: str = os.path.join(DATA_DIR, "hg19ToHg38.over.chain.gz"),
) -> Tuple[List[str], List[str]]:
    """
    Given a list of intervals in format chr:start-stop, lift them over acccording to the chain file
    and return the new coordinates, as well as those that were unmapped.
    This does NOT reorder the regions
    >>> liftover_intervals(["chr1:10134-10369", "chr1:804533-805145"])
    (['chr1:10134-10369', 'chr1:869153-869765'], [])
    >>> liftover_intervals(["chr1:804533-805145", "chr1:10134-10369"])
    (['chr1:869153-869765', 'chr1:10134-10369'], [])
    """

    assert os.path.isfile(chain_file), f"Cannot find chain file: {chain_file}"
    liftover_binary = shutil.which("liftOver")
    assert liftover_binary, "Cannot find liftover binary"

    # Write to a temporary file, pass that temporary file into liftover, read output
    tmp_id = random.randint(1, 10000)

    tmp_fname = f"liftover_intermediate_{tmp_id}.txt"
    tmp_out_fname = f"liftover_output_{tmp_id}.txt"
    tmp_unmapped_fname = f"liftover_unmapped_{tmp_id}.txt"

    with open(tmp_fname, "w") as sink:
        sink.write("\n".join(intervals) + "\n")

    cmd = f"{liftover_binary} {tmp_fname} {chain_file} {tmp_out_fname} {tmp_unmapped_fname} -positions"
    retcode = subprocess.call(shlex.split(cmd))
    assert retcode == 0, f"liftover exited with error code {retcode}"

    # Read in the output
    with open(tmp_out_fname) as source:
        retval = [l.strip() for l in source]
    with open(tmp_unmapped_fname) as source:
        unmapped = [l.strip() for l in source if not l.startswith("#")]
    assert len(retval) + len(unmapped) == len(intervals)

    if unmapped:
        logging.warning(f"Found unmapped regions: {len(unmapped)}")

    # Remove temporary files
    os.remove(tmp_fname)
    os.remove(tmp_out_fname)
    os.remove(tmp_unmapped_fname)
    # Fix the leftover intermediate file
    # This cannot be run in parallel mode
    for fname in glob.glob(f"liftOver_{platform.node()}_*.bedmapped"):
        os.remove(fname)
    for fname in glob.glob(f"liftOver_{platform.node()}_*.bedunmapped"):
        os.remove(fname)
    for fname in glob.glob(f"liftOver_{platform.node()}_*.bed"):
        os.remove(fname)

    return retval, unmapped



def liftover_atac_adata(
    adata: AnnData, chain_file: str = os.path.join(DATA_DIR, "hg19ToHg38.over.chain.gz")
) -> AnnData:
    """
    Lifts over the ATAC bins
    """
    lifted_var_names, unmapped_var_names = liftover_intervals(
        list(adata.var_names), chain_file=chain_file
    )
    keep_var_names = [n for n in adata.var_names if n not in unmapped_var_names]
    adata_trimmed = adata[:, keep_var_names]
    adata_trimmed.var_names = lifted_var_names
    adata_trimmed.var['features'] = adata_trimmed.var_names 
    adata_trimmed.var['genome'] = "GRCh38"
    return adata_trimmed



def aggregate_adata(fnames, feature_type, liftover):

    parsed = [get_ad_reader(fname, feature_type) for fname in fnames]

    if feature_type =="Peaks":
        if liftover:
            atac_liftover_parsed = [
                liftover_atac_adata(atac) if atac.var['genome'][0]=="hg19" else atac for atac in parsed
            ]
        else:
            atac_liftover_parsed = parsed
        if len(atac_liftover_parsed) > 1:
            atac_bins = harmonize_atac_intervals(
                atac_liftover_parsed[0].var_names, atac_liftover_parsed[1].var_names
            )
            for bins in atac_liftover_parsed[2:]:
                atac_bins = harmonize_atac_intervals(
                    atac_bins, bins.var_names
                )
            logging.info(f"Aggregated {len(atac_bins)} bins")
        else:
            atac_bins = list(atac_liftover_parsed[0].var_names)

        atac_parsed = [repool_atac_bins(p, atac_bins) for p in atac_liftover_parsed]
        parsed = atac_parsed
    idx = 1 
    for fname, p in zip(fnames, parsed):  # Make variable names unique and ensure sparse
        p.var_names_make_unique()
        p.obs_names_make_unique()
        p.X = scipy.sparse.csr_matrix(p.X)
        p.obs["source_file"] = fname
        prefix = f"{idx}_"
        p.obs_names = [f"{prefix}{obs_name}" for obs_name in p.obs_names]
        idx += 1
        print(p.obs)
    retval = parsed[0]
    if len(parsed) > 1:
         retval = retval.concatenate(*parsed[1:], join="inner")
    logging.info(f"{feature_type} data {retval.shape} (obs x var)")

    return retval



@functools.lru_cache(maxsize=2, typed=True)
def read_gtf_gene_to_pos(
    fname: str = None,
    acceptable_types: List[str] = None,
    addtl_attr_filters: dict = None,
    extend_upstream: int = 0,
    extend_downstream: int = 0,
) -> Dict[str, Tuple[str, int, int]]:
    """
    Given a gtf file, read it in and return as a ordered dictionary mapping genes to genomic ranges
    Ordering is done by chromosome then by position
    """
    # https://uswest.ensembl.org/info/website/upload/gff.html
    gene_to_positions = collections.defaultdict(list)
    gene_to_chroms = collections.defaultdict(set)

    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if line.startswith(b"#"):
                continue
            line = line.decode()
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [t.strip().split(" ", 1) for t in attrs.strip().split(";") if t]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            gene = attr_dict["gene_name"].strip('"')
            start = int(start)
            end = int(end)
            assert (
                start <= end
            ), f"Start {start} is not less than end {end} for {gene} with strand {strand}"
            if extend_upstream:
                if strand == "+":
                    start -= extend_upstream
                else:
                    end += extend_upstream
            if extend_downstream:
                if strand == "+":
                    end += extend_downstream
                else:
                    start -= extend_downstream

            gene_to_positions[gene].append(start)
            gene_to_positions[gene].append(end)
            gene_to_chroms[gene].add(chrom)

    slist = sortedcontainers.SortedList()
    for gene, chroms in gene_to_chroms.items():
        if len(chroms) != 1:
            logging.warn(
                f"Got multiple chromosomes for gene {gene}: {chroms}, skipping"
            )
            continue
        positions = gene_to_positions[gene]
        t = (chroms.pop(), min(positions), max(positions), gene)
        slist.add(t)

    retval = collections.OrderedDict()
    for chrom, start, stop, gene in slist:
        retval[gene] = (chrom, start, stop)
    return retval

def reorder_genes_by_chrom(
    genes, gtf_file=None, return_genes=False, return_chrom=False
):
    """Reorders list of genes by their genomic coordinate in the given gtf"""
    assert len(genes) > 0, "Got empty set of genes"
    genes_set = set(genes)
    genes_list = list(genes)
    assert len(genes_set) == len(genes), f"Got duplicates in genes"

    genes_to_pos = read_gtf_gene_to_pos(gtf_file)
    genes_intersection = [
        g for g in genes_to_pos if g in genes_set
    ]  # In order of position
    assert genes_intersection, "Got empty list of intersected genes"
    logging.info(f"{len(genes_intersection)} genes with known positions")
    genes_to_idx = {}
    for i, g in enumerate(genes_intersection):
        genes_to_idx[g] = i  # Record position of each gene in the ordered list

    slist = sortedcontainers.SortedList()  # Insert into a sorted list
    skip_count = 0
    for gene in genes_intersection:
        slist.add((genes_to_idx[gene], gene))

    genes_reordered = [g for _i, g in slist]
    if return_genes:  # Return the genes themselves in order
        retval = genes_reordered
    else:  # Return the indices needed to rearrange the genes in order
        retval = np.array([genes_list.index(gene) for gene in genes_reordered])
    chroms = [genes_to_pos[g][0] for _i, g in slist]
    assert len(chroms) == len(retval)

    if return_chrom:
        retval = (retval, chroms)
    return retval

def get_chrom_from_intervals(intervals: List[str], strip_chr: bool = True):
    """
    Given a list of intervals, return a list of chromosomes that those are on

    >>> get_chrom_from_intervals(['chr2:100-200', 'chr3:100-222'])
    ['2', '3']
    """
    retval = [interval.split(":")[0].strip() for interval in intervals]
    if strip_chr:
        retval = [chrom.strip("chr") for chrom in retval]
    return retval

def get_chrom_from_genes(genes: List[str], gtf_file=None):
    """
    Given a list of genes, return a list of chromosomes that those genes are on
    For missing: NA
    """
    gene_to_pos = read_gtf_gene_to_pos(gtf_file)
    retval = [gene_to_pos[gene][0] if gene in gene_to_pos else "NA" for gene in genes]
    return retval

def annotate_chroms(parsed, gtf_file: str = "") -> None:
    """Annotates chromosome information on the var field, without the chr prefix"""
    # gtf_file can be empty if we're using atac intervals
    feature_chroms = (
        get_chrom_from_intervals(parsed.var_names)
        if list(parsed.var_names)[0].startswith("chr")
        else get_chrom_from_genes(parsed.var_names, gtf_file)
    )
    parsed.var["chrom"] = feature_chroms
    
    return parsed

def is_numeric(x) -> bool:
    """Return True if x is numeric"""
    try:
        x = float(x)
        return True
    except ValueError:
        return False


def annotate_basic_adata_metrics(adata: AnnData) -> None:
    """Annotate with some basic metrics"""
    assert isinstance(adata, AnnData)
    adata.obs["n_counts"] = np.squeeze(np.asarray((adata.X.sum(1))))
    adata.obs["log1p_counts"] = np.log1p(adata.obs["n_counts"])
    adata.obs["n_genes"] = np.squeeze(np.asarray(((adata.X > 0).sum(1))))

    adata.var["n_counts"] = np.squeeze(np.asarray(adata.X.sum(0)))
    adata.var["log1p_counts"] = np.log1p(adata.var["n_counts"])
    adata.var["n_cells"] = np.squeeze(np.asarray((adata.X > 0).sum(0)))
    
    return adata


def filter_adata_cells_and_genes(
    x: AnnData,
    filter_cell_min_counts=None,
    filter_cell_max_counts=None,
    filter_cell_min_genes=None,
    filter_cell_max_genes=None,
    filter_gene_min_counts=None,
    filter_gene_max_counts=None,
    filter_gene_min_cells=None,
    filter_gene_max_cells=None,
) -> None:
    """Filter the count table in place given the parameters based on actual data"""
    args = locals()
    filtering_cells = any(
        [args[arg] is not None for arg in args if arg.startswith("filter_cell")]
    )
    filtering_genes = any(
        [args[arg] is not None for arg in args if arg.startswith("filter_gene")]
    )

    def ensure_count(value, max_value) -> int:
        """Ensure that the value is a count, optionally scaling to be so"""
        if value is None:
            return value  # Pass through None
        retval = value
        if isinstance(value, float):
            assert 0.0 < value < 1.0
            retval = int(round(value * max_value))
        assert isinstance(retval, int)
        return retval

    assert isinstance(x, AnnData)
    # Perform filtering on cells
    if filtering_cells:
        logging.info(f"Filtering {x.n_obs} cells")
    if filter_cell_min_counts is not None:
        sc.pp.filter_cells(
            x,
            min_counts=ensure_count(
                filter_cell_min_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after min count: {x.n_obs}")
    if filter_cell_max_counts is not None:
        sc.pp.filter_cells(
            x,
            max_counts=ensure_count(
                filter_cell_max_counts, max_value=np.max(x.obs["n_counts"])
            ),
        )
        logging.info(f"Remaining cells after max count: {x.n_obs}")
    if filter_cell_min_genes is not None:
        sc.pp.filter_cells(
            x,
            min_genes=ensure_count(
                filter_cell_min_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after min genes: {x.n_obs}")
    if filter_cell_max_genes is not None:
        sc.pp.filter_cells(
            x,
            max_genes=ensure_count(
                filter_cell_max_genes, max_value=np.max(x.obs["n_genes"])
            ),
        )
        logging.info(f"Remaining cells after max genes: {x.n_obs}")

    # Perform filtering on genes
    if filtering_genes:
        logging.info(f"Filtering {x.n_vars} vars")
    if filter_gene_min_counts is not None:
        sc.pp.filter_genes(
            x,
            min_counts=ensure_count(
                filter_gene_min_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after min count: {x.n_vars}")
    if filter_gene_max_counts is not None:
        sc.pp.filter_genes(
            x,
            max_counts=ensure_count(
                filter_gene_max_counts, max_value=np.max(x.var["n_counts"])
            ),
        )
        logging.info(f"Remaining vars after max count: {x.n_vars}")
    if filter_gene_min_cells is not None:
        sc.pp.filter_genes(
            x,
            min_cells=ensure_count(
                filter_gene_min_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after min cells: {x.n_vars}")
    if filter_gene_max_cells is not None:
        sc.pp.filter_genes(
            x,
            max_cells=ensure_count(
                filter_gene_max_cells, max_value=np.max(x.var["n_cells"])
            ),
        )
        logging.info(f"Remaining vars after max cells: {x.n_vars}")

    return x

def normalize_count_table(
    x: AnnData,
    size_factors: bool = True,
    log_trans: bool = True,
    normalize: bool = True,
) -> AnnData:
    """
    Normalize the count table using method described in DCA paper, performing operations IN PLACE
    rows correspond to cells, columns correspond to genes (n_obs x n_vars)
    s_i is the size factor per cell, total number of counts per cell divided by median of total counts per cell
    x_norm = zscore(log(diag(s_i)^-1 X + 1))

    Reference:
    https://github.com/theislab/dca/blob/master/dca/io.py

    size_factors - calculate and normalize by size factors
    top_n - retain only the top n features with largest variance after size factor normalization
    normalize - zero mean and unit variance
    log_trans - log1p scale data
    """
    assert isinstance(x, AnnData)
    if log_trans or size_factors or normalize:
        x.raw = x.copy()  # Store the original counts as .raw
    # else:
    #     x.raw = x
    if size_factors:
        logging.info("Computing size factors")
        n_counts = np.squeeze(
            np.array(x.X.sum(axis=1))
        )  # Number of total counts per cell
        # Normalizes each cell to total count equal to the median of total counts pre-normalization
        sc.pp.normalize_total(x, inplace=True)
        # The normalized values multiplied by the size factors give the original counts
        x.obs["size_factors"] = n_counts / np.median(n_counts)
        x.uns["median_counts"] = np.median(n_counts)
        logging.info(f"Found median counts of {x.uns['median_counts']}")
        logging.info(f"Found maximum counts of {np.max(n_counts)}")
    else:
        x.obs["size_factors"] = 1.0
        x.uns["median_counts"] = 1.0
    
    data_matrix = x.X.todense()

    # Check if there are any negative values in the data matrix
    has_negative_values = np.any(data_matrix < 0)

   
    if log_trans:  # Natural logrithm
        logging.info("Log transforming data")
        sc.pp.log1p(
            x,
            chunked=True,
            copy=False,
            chunk_size=100000,
        )
    data_matrix = x.X.todense()

    # Check if there are any negative values in the data matrix
    has_negative_values = np.any(data_matrix < 0)

    # if zero_center True, then remove x.X = x.X.toarray() and data_matrix = x
    # if zero_center False, then keep x.X = x.X.toarray() and data_matrix = x.X.todense()

    if normalize:
        logging.info("Normalizing data to zero mean unit variance")
        sc.pp.scale(x, zero_center=False, copy=False)

    data_matrix = x.X.todense()    # Check if there are any negative values in the data matrix
    has_negative_values2 = np.any(data_matrix < 0)
   
    x.X = x.X.toarray()

    return x

def size_norm_counts(x: AnnData) -> AnnData:
    """Computes and stores table of normalized counts w/ size factor adjustment and no other normalization"""
    logging.info(f"Setting size normalized counts")
    # raw_counts_anndata = AnnData(
    #     scipy.sparse.csr_matrix(x.raw.X),
    #     obs=pd.DataFrame(index=x.obs_names),
    #     var=pd.DataFrame(index=x.var_names),
    # )
    sc.pp.normalize_total(x, inplace=True)
    return x

def split_train_valid_test(x: AnnData) -> Dict[str, List[int]]:
    """ Splits the dataset into train/valid/test """
    logging.info("Random splitting the dataset with ratio 0.7/0.15/0.15")
   
    # Define the proportions for train, validation, and test sets
    train_ratio = 0.7
    valid_ratio = 0.15
    test_ratio = 0.15

    # Shuffle the indices of the dataset
    num_samples = x.shape[0]
    indices = np.random.permutation(num_samples)


    # Split the indices according to the defined ratios
    train_end = int(train_ratio * num_samples)
    valid_end = int((train_ratio + valid_ratio) * num_samples)

    train_idx = indices[:train_end]
    valid_idx = indices[train_end:valid_end]
    valid_idx = indices[valid_end:]

    data_split_idx = {}
    data_split_idx["train"] = train_idx
    data_split_idx["valid"] = valid_idx
    data_split_idx["test"] = valid_idx
    # # Create the train, validation, and test sets
    # train_set = data[train_indices]
    # valid_set = data[valid_indices]
    # test_set = data[test_indices]


    return data_split_idx




