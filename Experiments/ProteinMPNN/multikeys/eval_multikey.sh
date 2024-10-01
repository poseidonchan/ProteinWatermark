#!/bin/bash
#SBATCH --job-name=multikey                                 
#SBATCH --output=experiment.log%j                              
#SBATCH --error=experiment.err.%j                               
#SBATCH --time=24:00:00                                         
#SBATCH --partition=cbcb-heng
#SBATCH --account=cbcb-heng
#SBATCH --qos=highmem                                           
#SBATCH --nodes=1                                               
#SBATCH --ntasks=1                                              
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb                                               

# Accept start_key and end_key as command-line arguments
start_key=$1
end_key=$2

source /fs/cbcb-scratch/cys/miniconda3/etc/profile.d/conda.sh
source /nfshomes/cys/.bashrc
cd /fs/cbcb-scratch/cys/protein_watermark/ProteinMPNN/multikeys
conda activate esmfold

# Run the multikeys.py script with the specified key range
python multikeys.py --start_key ${start_key} --end_key ${end_key}