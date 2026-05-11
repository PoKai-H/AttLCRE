#!/bin/bash

GRID_BASE="data_experiment/multiwoz_generator_experiment/grid"
OUTPUT_BASE="experiment_outputs"

mkdir -p ${OUTPUT_BASE}

declare -A CHECKPOINTS
CHECKPOINTS[bert]="data/outputs_experiment/bert/best_model.pt"
CHECKPOINTS[longformer]="data/outputs_experiment/longformer/best_model.pt"

for model in bert longformer
do
  for d in 20 50 100 200
  do
    for den in 1 5 10 20
    do
      for s in 1 2 3
      do
        echo "Running model=$model distance=$d density=$den seed=$s"

        python3 run_test_only.py \
          --model ${model} \
          --test_path ${GRID_BASE}/d${d}_den${den}_seed${s}/test.json \
          --checkpoint_path ${CHECKPOINTS[$model]} \
          --output_dir ${OUTPUT_BASE}/${model}/d${d}_den${den}_seed${s} \
          --render_mode full \
          --include_speaker
      done
    done
  done
done

python3 heatmap.py