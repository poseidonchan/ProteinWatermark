#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=protein_jupyter                                   # sets the job name
#SBATCH --output=notebook.out.%j                              # indicates a file to redirect STDOUT to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=notebook.out.%j                               # indicates a file to redirect STDERR to; %j is the jobid. Must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=24:00:00                                         # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --partition=cbcb
#SBATCH --account=cbcb
#SBATCH --qos=highmem                                           # set QOS, this will determine what resources can be requested
#SBATCH --nodes=1                                               # number of nodes to allocate for your job
#SBATCH --ntasks=1                                              # request 4 cpu cores be reserved for your node total
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=64gb                                               # memory required by job; if unit is not specified MB will be assumed

source /fs/cbcb-scratch/cys/miniconda3/etc/profile.d/conda.sh
source /nfshomes/cys/.bashrc
cd /fs/cbcb-scratch/cys/protein_watermark/ProteinMPNN/wm_examples
conda activate esmfold
folder_with_pdbs="../inputs/sampled_monomers/"

output_dir="../wm_outputs/monomer_wm_0.7"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

path_for_parsed_chains=$output_dir"/parsed_pdbs.jsonl"

python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

python ../protein_mpnn_run_wm.py \
        --jsonl_path $path_for_parsed_chains \
        --out_folder $output_dir \
        --num_seq_per_target 50 \
        --sampling_temp "0.7" \
        --seed 37 \
        --batch_size 1
