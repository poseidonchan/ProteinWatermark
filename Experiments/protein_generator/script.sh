echo "Running binder design..."

nohup python ./inference.py \
    --num_designs 50 \
    --out examples/out/binder_design \
    --pdb examples/pdbs/cd86.pdb \
    --T 25 --save_best_plddt \
    --contigs B1-110,0 25-75 \
    --hotspots B40,B32,B87,B96,B30 > binder_original.log 2>&1 &
    
wait

nohup python ./inference_wm.py \
    --num_designs 50 \
    --out examples/out/binder_design \
    --pdb examples/pdbs/cd86.pdb \
    --T 25 --save_best_plddt \
    --contigs B1-110,0 25-75 \
    --hotspots B40,B32,B87,B96,B30 > binder_wm.log 2>&1 &
    
wait

echo "Running motif design..."

nohup python ./inference.py \
    --num_designs 50 \
    --out examples/out/design \
    --pdb examples/pdbs/rsv5_5tpn.pdb \
    --contigs 0-25,A163-181,25-30 --T 25 --save_best_plddt > motif_original.log 2>&1 &

wait

nohup python ./inference_wm.py \
    --num_designs 50 \
    --out examples/out/design \
    --pdb examples/pdbs/rsv5_5tpn.pdb \
    --contigs 0-25,A163-181,25-30 --T 25 --save_best_plddt > motif_wm.log 2>&1 &

wait

echo "Running sequence conditioned design..."

nohup python ./inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --sequence XXXXXXXXXXXXXXXXPEPSEQXXXXXXXXXXXXXXXX \
    --T 25 --save_best_plddt > sequence_conditioned_original.log 2>&1 &

wait

nohup python ./inference_wm.py \
    --num_designs 10 \
    --out examples/out/design \
    --sequence XXXXXXXXXXXXXXXXPEPSEQXXXXXXXXXXXXXXXX \
    --T 25 --save_best_plddt > sequence_conditioned_wm.log 2>&1 &

wait

echo "Running secondary structure conditioned design..."

nohup python ./inference.py \
    --num_designs 10 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --secondary_structure XXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXX > secondary_structure_original.log

wait

nohup python ./inference_wm.py \
    --num_designs 10 \
    --out examples/out/design \
    --contigs 100 \
    --T 25 --save_best_plddt \
    --secondary_structure XXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXXXXXXXHHHHXXXLLLXXXXX > secondary_structure_wm.log

echo "ALL TASK FINISHED"
