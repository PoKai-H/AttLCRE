## checkpoint 
outputs/bert/best_model.pt

## new tests
data/new_test.json

```bash
python run_test_only.py \
    --model bert \
    --test_path new_data/test_hard.json \
    --checkpoint_path outputs/bert/best_model.pt \
    --output_dir outputs/new_test_eval
```