#!/bin/bash

for i in {0..9}
do
    start=$((i * 100 + 1))
    end=$(( (i + 1) * 100 ))

    # Submit the job with the specified key range
    sbatch --job-name=multikey_${start}_${end} eval_multikey.sh ${start} ${end}
done