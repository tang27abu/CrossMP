# CrossMP

CrossMP is a Python-based deep learning model designed to translation between scRAN-seq and scATAC-seq.

## Installation

Install CrossMP environment using `conda`:

```bash
conda env create -f environment_crossmp.yml
conda activate crossmp
```

## Usage

### Preprocess
Preprocess the scATAC-seq and scRNA-seq data by running:
```bash
python preprocess/preprocessing.py --atac ATAC.h5ad --rna RNA.h5ad --organism human --outdir output_dir
```
Generating the gene activity scores after completing the scATAC-seq preprocessing:
```bash
Rscript preprocess/GeneActivityScore.R  output_dir
```
Preprocess the gene activity scores by running:
```bash
python preprocess/preprocessing_rp.py --organism human --datadir output_dir 
```

### Train & evaluate model

```bash
conda activate crossmp
python train.py  --outdir model_out --datadir data_dir
```
Note that the data_dir is the output_dir used during preprocessing.

