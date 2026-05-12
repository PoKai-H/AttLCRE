#!/bin/bash

GRID_BASE="data_experiment/multiwoz_generator_experiment/grid"
OUTPUT_BASE="data_experiment/experiment_output"

BERT_CKPT="data_experiment/bert/best_model.pt"
LONGFORMER_CKPT="data_experiment/longformer/best_model.pt"

#for model in bert longformer
for model in longformer
do
  if [ "$model" = "bert" ]; then
    CHECKPOINT=$BERT_CKPT
  elif [ "$model" = "longformer" ]; then
    CHECKPOINT=$LONGFORMER_CKPT
  fi

  if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found for $model: $CHECKPOINT"
    continue
  fi

  mkdir -p ${OUTPUT_BASE}/${model}

  for d in 50 100 200
  do
    for den in 1 5 10 20
    do
      for seed in 1 2 3
      do
        echo "Running $model | distance=$d | density=$den | seed=$seed"

        python3 run_test_only.py \
          --model ${model} \
          --test_path ${GRID_BASE}/d${d}_den${den}_seed${seed}/test.json \
          --checkpoint_path ${CHECKPOINT} \
          --output_dir ${OUTPUT_BASE}/${model}/d${d}_den${den}_seed${seed} \
          --render_mode full \
          --include_speaker
      done
    done
  done
done

python3 heatmap.py