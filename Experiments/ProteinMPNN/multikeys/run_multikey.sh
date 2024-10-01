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
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32gb                                               



source /fs/cbcb-scratch/cys/miniconda3/etc/profile.d/conda.sh
source /nfshomes/cys/.bashrc
cd /fs/cbcb-scratch/cys/protein_watermark/ProteinMPNN/multikeys
conda activate esmfold

# Define the paths
utils_wm_file="../protein_mpnn_utils_wm.py"
design_script="../protein_mpnn_run_wm.py"
path_for_parsed_chains="./parsed_pdbs.jsonl"

parsed_chains="$path_for_parsed_chains"
output_dir_base="./design_outputs/"

folder_with_pdbs="./pdb/"


python ../helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

# Loop over keys from 1 to 1000
for key in {1..1000}
do
    echo "Running design with private key: $key"

    # Update the private key in protein_mpnn_utils_wm.py
    sed -i '/delta_wp = WatermarkLogitsProcessor(/{n;s/b".*"/b"'"$key"'"/;}' "$utils_wm_file"

    # Define the output directory for the current key
    output_dir="${output_dir_base}/key_${key}"
    mkdir -p "$output_dir"

    # Run the design script
    python "$design_script" \
        --jsonl_path "$path_for_parsed_chains" \
        --out_folder "$output_dir" \
        --num_seq_per_target 10 \
        --sampling_temp "0.5" \
        --seed 37 \
        --batch_size 1
done