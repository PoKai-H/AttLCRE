#!/bin/bash

GRID_BASE=data_experiment/grid
OUTPUT_BASE=experiment_outputs

BERT_CHECKPOINT=data/outputs_experiment/bert/best_model.pt
LONGFORMER_CHECKPOINT=data/outputs_experiment/longformer/best_model.pt

echo "Running BERT attention analysis..."

python3 analyze_attention_grid.py \
  --model bert \
  --checkpoint_path ${BERT_CHECKPOINT} \
  --grid_base ${GRID_BASE} \
  --output_base ${OUTPUT_BASE} \
  --max_length 512

echo "Running Longformer attention analysis..."

python3 analyze_attention_grid.py \
  --model longformer \
  --checkpoint_path ${LONGFORMER_CHECKPOINT} \
  --grid_base ${GRID_BASE} \
  --output_base ${OUTPUT_BASE} \
  --max_length 1024

echo "All attention analysis completed."