#!/bin/bash
folder_with_pdbs="./inputs/sampled_monomers/"

output_dir="./wm_outputs/monomer_design_entropy"
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi

path_for_parsed_chains="${output_dir}/parsed_pdbs.jsonl"

python ./helper_scripts/parse_multiple_chains.py --input_path="$folder_with_pdbs" --output_path="$path_for_parsed_chains"

# Define an array of sampling temperatures
sampling_temps=("0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1")

# Iterate over sampling temperatures
for temp in "${sampling_temps[@]}"; do
    # Construct the temperature-specific output directory path
    temp_output_dir="${output_dir}/${temp}"
    # Create the directory if it doesn't exist
    if [ ! -d "$temp_output_dir" ]; then
        mkdir -p "$temp_output_dir"
    fi

    python ./protein_mpnn_run_wm.py \
        --jsonl_path "$path_for_parsed_chains" \
        --out_folder "$temp_output_dir" \
        --num_seq_per_target 1 \
        --sampling_temp "$temp" \
        --seed 37 \
        --batch_size 1
done
