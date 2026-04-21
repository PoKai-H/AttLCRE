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
  --output_dir new_data2/outputs/bert/test_full \
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

### Oracle role ablation
```bash
python run_test_only.py \
  --model bert \
  --test_path new_data2/test.json \
  --checkpoint_path new_data2/outputs/bert/best_model.pt \
  --output_dir new_data2/outputs/bert/test_candidate \
  --render_mode full\
  --include_speaker
  --include_roles
```