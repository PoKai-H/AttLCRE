## checkpoint 
new_data2/outputs/bert/best_model.pt

## new tests
new_data2/test.json

# Training

```bash
python run.py --model bert 
```

# Testing Only
### Full Context
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/test.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/test_full \
  --render_mode full \
  --include_speaker
```

### Remove Signal
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/rm_signal.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/test_rm_signal \
  --render_mode full \
  --include_speaker
```


### Local Only (4 nearest sentence)
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/test.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/test_local \
  --render_mode local_only \
  --local_k 4 \
  --include_speaker
```

### Candidate Only
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/test.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/test_candidate \
  --render_mode candidate_only \
  --include_speaker
```

## Control Distance Test

The distance here are not tokens but turns, average number of tokens in one turn is 10

### Short distance 
signal distance: min 1, avg 3.17, max 6
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/short_distance.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/test_short_distance \
  --render_mode full \
  --include_speaker
```

### Long distance
signal distance: min 21, avg 28.07, max 36
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/long_distance.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/test_long_distance \
  --render_mode full \
  --include_speaker
```

### High distractor
signal distance: min 21, avg 30.23, max 40
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/high_distractor.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/high_distractor \
  --render_mode full \
  --include_speaker
```

