#!/bin/bash
#SBATCH --job-name=analyze_multikey                                 
#SBATCH --output=experiment.log%j                              
#SBATCH --error=experiment.err.%j                               
#SBATCH --time=24:00:00                                         
#SBATCH --partition=cbcb-heng
#SBATCH --account=cbcb-heng
#SBATCH --qos=highmem                                           
#SBATCH --nodes=1                                               
#SBATCH --ntasks=1                                              
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb                                               



source /fs/cbcb-scratch/cys/miniconda3/etc/profile.d/conda.sh
source /nfshomes/cys/.bashrc
cd /fs/cbcb-scratch/cys/protein_watermark/ProteinMPNN/multikeys
conda activate esmfold

python analyze.py