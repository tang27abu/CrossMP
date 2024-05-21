## conda activate MAESTRO
## mamba deactivate
# $ conda config --add channels defaults
# $ conda config --add channels liulab-dfci
# $ conda config --add channels bioconda
# $ conda config --add channels conda-forge
# # To make the installation faster, we recommend using mamba
# $ conda install mamba -c conda-forge
# $ mamba create -n MAESTRO maestro=1.5.1 -c liulab-dfci
# # Activate the environment
# $ conda activate MAESTRO
#pip install tables==3.7.0

library(reticulate)
use_python("/scratch/zl7w2/tools/miniconda3/envs/MAESTRO/bin/python3", required = TRUE)
library(anndata)
library(logr)
library(MAESTRO)
library(Matrix)




##################################################
# Load H5AD File
##################################################

loadData <- function(h5ad_f) {
 
  log_print(paste0("Loading the ", h5ad_f))
  data <- read_h5ad(h5ad_f)
  obs_len <- dim(data$X)[1]
  var_len <-dim(data$X)[2]
  log_print(paste0("Sparse matrix size : ", obs_len, " obs X ", var_len, " vars"))
  return(data)
}


##################################################
# Get Gene Activity Score
##################################################

GeneActivity <- function(data, chunk_size) {
  
  log_print(paste0("chunk size: ", chunk_size))
  obs_len <- dim(data$X)[1]
  idx_col <- seq(1, obs_len, by=chunk_size)
  idx <- 0

  for(i in idx_col){
    j <- min(c(i+chunk_size-1, obs_len)) 

    dataP = t(data$X[i:j,])
    log_print(paste0("Start processing matrix from cell ",i," to cell " ,j, ""))
    # dataG <- ATACCalculateGenescore(dataP, organism = "GRCh38", decaydistance = 10000, model = "Enhanced")
    dataG <- ATACCalculateGenescore(dataP, organism = "GRCm38", decaydistance = 10000, model = "Enhanced")
    if(idx == 0){
      merged_spm <- dataG
    }else{
      tmp <- merged_spm
      merged_spm  <- cbind(tmp,dataG)

    }
    #log_print(paste0("TTTGTTGGTTGGTTAG-1-4: ", merged_spm[,'TTTGTTGGTTGGTTAG-1-4']))
    idx <- idx + 1
  }

  ad <- AnnData(
    X = t(merged_spm),
    obs = data.frame(row.names = rownames(t(merged_spm))),
    var = data.frame(row.names = colnames(t(merged_spm)))
  )

  return (ad)
}



##################################################
# Write H5AD File
##################################################

WriteH5ad <- function(data, filename) {


  log_print(paste0("Writing to h5ad file ", filename))
  write_h5ad(data, filename)

}



args <- commandArgs(trailingOnly = TRUE)
print(args)
data_dir <- args[1]

# data_dir <- "/scratch/zl7w2/data/test/"

st <- format(Sys.time(), "%Y%m%d%H%M%S")
tmp <- file.path(paste0("maestro_",st,".log"))
lf <- log_open(tmp)

data <- loadData(file.path(data_dir, "hc_atac.h5ad"))


chunk_size <- 5000
annD <- GeneActivity(data, chunk_size)

WriteH5ad(annD, file.path(data_dir, "hc_rp.h5ad"))

writeLines(readLines(lf))
log_close()

