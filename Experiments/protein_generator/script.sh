echo "Running binder design..."

nohup python ./inference.py \
    --num_designs 500 \
    --out examples/out/binder_design \
    --pdb examples/pdbs/cd86.pdb \
    --T 25 --save_best_plddt \
    --contigs B1-110,0 25-75 \
    --hotspots B40,B32,B87,B96,B30 > binder_original.log 2>&1 &
    
wait

nohup python ./inference_multinomial.py \
    --num_designs 500 \
    --out examples/out/binder_design \
    --pdb examples/pdbs/cd86.pdb \
    --T 25 --save_best_plddt \
    --contigs B1-110,0 25-75 \
    --hotspots B40,B32,B87,B96,B30 > binder_multinomial.log 2>&1 &
    
wait

nohup python ./inference_multinomial_wm.py \
    --num_designs 500 \
    --out examples/out/binder_design \
    --pdb examples/pdbs/cd86.pdb \
    --T 25 --save_best_plddt \
    --contigs B1-110,0 25-75 \
    --hotspots B40,B32,B87,B96,B30 > binder_wm.log 2>&1 &
    
wait


echo "Running secondary structure conditioned design..."

nohup python ./inference.py \
    --num_designs 500 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --secondary_structure XXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXX > secondary_structure_original.log

wait

nohup python ./inference_multinomial.py \
    --num_designs 500 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --secondary_structure XXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXX > secondary_structure_multinomial.log

nohup python ./inference_multinomial_wm.py \
    --num_designs 500 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --secondary_structure XXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXX > secondary_structure_wm.log

echo "ALL TASK FINISHED"
