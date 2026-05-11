# Exploratory Improvement

To better understand the causes of retrieval failure, we conduct exploratory interventions primarily on the BERT baseline. We choose BERT because its failures under our controlled benchmark are more pronounced and easier to interpret. In contrast, architectures such as Longformer introduce sparse-attention mechanisms that may partially alleviate long-context scaling issues, making it more difficult to isolate the underlying retrieval behaviors.

Importantly, our goal is not to outperform all long-context architectures. Instead, we use controlled architectural interventions to probe possible causes of retrieval failure under sparse long-horizon settings. In particular, we explore two complementary hypotheses: (1) retrieval failures may arise from insufficient global aggregation capacity, and (2) attention mechanisms may fail to allocate sufficient focus toward sparse task-relevant signals in distractor-heavy contexts.

To investigate these hypotheses, we evaluate two exploratory modifications on the BERT baseline: explicit memory tokens and attention biasing mechanisms.


## 1. Memory-Enhanced BERT
One limitation of using the standard [CLS] token for sequence classification is that the entire input sequence must be compressed into a single global representation. In long-context settings with sparse relevant signals and many distractors, this compression may be insufficient, especially when the relevant evidence appears far from the query.

To test whether explicit aggregation slots can improve long-range selective retrieval, we introduce a small number of learnable memory tokens at the beginning of the input sequence. These memory tokens are trained jointly with the model and can attend to all context tokens through self-attention. Instead of using only the final [CLS] representation for classification, we concatenate the [CLS] representation with the averaged representation of the memory tokens.

### Theoretical Intuition.

Let the input sequence be
$$
X = [x_1, x_2, \dots, x_n].
$$

We introduce $k$ learnable memory tokens
$$
M = [m_1, \dots, m_k],
$$
and prepend them to the original sequence:
$$
\tilde{X} = [m_1, \dots, m_k, x_1, \dots, x_n].
$$

Since memory tokens participate in self-attention together with the original sequence tokens, each memory token can aggregate information from the full context:
$$
m_i^{(l+1)}
=
\sum_{j=1}^{n+k}
\alpha_{ij}^{(l)}
W_V h_j^{(l)},
$$
where $$\alpha_{ij}^{(l)}$$ denotes the self-attention weight at layer $l$.

This allows the memory tokens to act as additional trainable aggregation slots beyond the standard \texttt{[CLS]} token. Intuitively, these memory representations may provide additional capacity for maintaining sparse and temporally distant task-relevant information under long-context settings.

### Result
#### Multiwoz
#### Synthetic
(csci5527) ➜  AttLCRE git:(main) ✗ python run.py --model bert_mem                                                                              
Loading data...
Rendering rows...
Model: bert_mem
Device: mps
Max length: 512
Train rows: 4000 | Val rows: 2000 | Test rows: 3564
Train batches: 500 | Val batches: 125 | Test batches: 223
Train batch size: 8 | Eval batch size: 16

Training setup | Epochs: 3 | Train batches/epoch: 500 | Val batches: 125 | Total steps: 1500 | Warmup steps: 150 | Device: mps

Starting epoch 1/3 (500 train batches)
Epoch 1/3 | Batch 1/500 | Loss: 0.7063 | Avg Loss: 0.7063 | LR: 1.33e-07
Epoch 1/3 | Batch 50/500 | Loss: 0.7108 | Avg Loss: 0.6100 | LR: 6.67e-06
Epoch 1/3 | Batch 100/500 | Loss: 0.9690 | Avg Loss: 0.6053 | LR: 1.33e-05
Epoch 1/3 | Batch 150/500 | Loss: 0.3582 | Avg Loss: 0.5797 | LR: 2.00e-05
Epoch 1/3 | Batch 200/500 | Loss: 0.7785 | Avg Loss: 0.5807 | LR: 1.93e-05
Epoch 1/3 | Batch 250/500 | Loss: 0.5653 | Avg Loss: 0.5768 | LR: 1.85e-05
Epoch 1/3 | Batch 300/500 | Loss: 0.6349 | Avg Loss: 0.5660 | LR: 1.78e-05
Epoch 1/3 | Batch 350/500 | Loss: 0.4254 | Avg Loss: 0.5704 | LR: 1.70e-05
Epoch 1/3 | Batch 400/500 | Loss: 0.3179 | Avg Loss: 0.5652 | LR: 1.63e-05
Epoch 1/3 | Batch 450/500 | Loss: 0.5250 | Avg Loss: 0.5625 | LR: 1.56e-05
Epoch 1/3 | Batch 500/500 | Loss: 0.3107 | Avg Loss: 0.5575 | LR: 1.48e-05
Evaluating epoch 1/3...
Epoch 1/3 | Train Loss: 0.5575 | Val Sample Acc: 0.5380
Saved new best model to: new_data2/outputs/bert_mem/best_model.pt

Starting epoch 2/3 (500 train batches)
Epoch 2/3 | Batch 1/500 | Loss: 0.3476 | Avg Loss: 0.3476 | LR: 1.48e-05
Epoch 2/3 | Batch 50/500 | Loss: 0.6058 | Avg Loss: 0.5752 | LR: 1.41e-05
Epoch 2/3 | Batch 100/500 | Loss: 0.4167 | Avg Loss: 0.5429 | LR: 1.33e-05
Epoch 2/3 | Batch 150/500 | Loss: 0.5165 | Avg Loss: 0.5303 | LR: 1.26e-05
Epoch 2/3 | Batch 200/500 | Loss: 0.5512 | Avg Loss: 0.5223 | LR: 1.19e-05
Epoch 2/3 | Batch 250/500 | Loss: 0.3249 | Avg Loss: 0.5103 | LR: 1.11e-05
Epoch 2/3 | Batch 300/500 | Loss: 0.8416 | Avg Loss: 0.5034 | LR: 1.04e-05
Epoch 2/3 | Batch 350/500 | Loss: 0.1803 | Avg Loss: 0.5014 | LR: 9.63e-06
Epoch 2/3 | Batch 400/500 | Loss: 0.4575 | Avg Loss: 0.4964 | LR: 8.89e-06
Epoch 2/3 | Batch 450/500 | Loss: 0.6311 | Avg Loss: 0.4966 | LR: 8.15e-06
Epoch 2/3 | Batch 500/500 | Loss: 0.3651 | Avg Loss: 0.4961 | LR: 7.41e-06
Evaluating epoch 2/3...
Epoch 2/3 | Train Loss: 0.4961 | Val Sample Acc: 0.6340
Saved new best model to: new_data2/outputs/bert_mem/best_model.pt

Starting epoch 3/3 (500 train batches)
Epoch 3/3 | Batch 1/500 | Loss: 0.8792 | Avg Loss: 0.8792 | LR: 7.39e-06
Epoch 3/3 | Batch 50/500 | Loss: 0.5190 | Avg Loss: 0.4710 | LR: 6.67e-06
Epoch 3/3 | Batch 100/500 | Loss: 0.3478 | Avg Loss: 0.4579 | LR: 5.93e-06
Epoch 3/3 | Batch 150/500 | Loss: 0.4938 | Avg Loss: 0.4688 | LR: 5.19e-06
Epoch 3/3 | Batch 200/500 | Loss: 0.4352 | Avg Loss: 0.4644 | LR: 4.44e-06
Epoch 3/3 | Batch 250/500 | Loss: 0.5049 | Avg Loss: 0.4669 | LR: 3.70e-06
Epoch 3/3 | Batch 300/500 | Loss: 0.1417 | Avg Loss: 0.4643 | LR: 2.96e-06
Epoch 3/3 | Batch 350/500 | Loss: 0.3898 | Avg Loss: 0.4564 | LR: 2.22e-06
Epoch 3/3 | Batch 400/500 | Loss: 0.3047 | Avg Loss: 0.4578 | LR: 1.48e-06
Epoch 3/3 | Batch 450/500 | Loss: 0.3111 | Avg Loss: 0.4572 | LR: 7.41e-07
Epoch 3/3 | Batch 500/500 | Loss: 0.2608 | Avg Loss: 0.4535 | LR: 0.00e+00
Evaluating epoch 3/3...
Epoch 3/3 | Train Loss: 0.4535 | Val Sample Acc: 0.6440
Saved new best model to: new_data2/outputs/bert_mem/best_model.pt

===== Validation =====
Sample-level accuracy: 0.6440

By difficulty:
{
  "hard": {
    "count": 500.0,
    "accuracy": 0.644
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.644
  }
}

===== Test =====
Sample-level accuracy: 0.7003

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.7003367003367004
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.7003367003367004
  }
}

Saved results to: new_data2/outputs/bert_mem

---
### Full Context
===== New Test Set =====
Sample-level accuracy: 0.7104

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.7104377104377104
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.7104377104377104
  }
}
Saved predictions to: new_data2/outputs/bert_mem/test_full/test_predictions.json

---
### Remove Signal
===== New Test Set =====
Sample-level accuracy: 0.2334

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2334455667789001
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2334455667789001
  }
}
Saved predictions to: new_data2/outputs/bert_mem/test_rm_signal/test_predictions.json

---
### Local Only (4 nearest sentence)
===== New Test Set =====
Sample-level accuracy: 0.2963

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2962962962962963
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2962962962962963
  }
}
Saved predictions to: new_data2/outputs/bert_mem/test_local/test_predictions.json

---
### Candidate Only
===== New Test Set =====
Sample-level accuracy: 0.2312

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.23120089786756454
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.23120089786756454
  }
}
Saved predictions to: new_data2/outputs/bert_mem/test_candidate/test_predictions.json

---
### Short Distance
===== New Test Set =====
Sample-level accuracy: 0.6400

By difficulty:
{
  "short_distance": {
    "count": 500.0,
    "accuracy": 0.64
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.64
  }
}
Saved predictions to: new_data2/outputs/bert_mem/test_short_distance/test_predictions.json

### Long Distance
===== New Test Set =====
Sample-level accuracy: 0.5880

By difficulty:
{
  "long_distance": {
    "count": 500.0,
    "accuracy": 0.588
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.588
  }
}
Saved predictions to: new_data2/outputs/bert_mem/test_long_distance/test_predictions.json


### High Distractor

===== New Test Set =====
Sample-level accuracy: 0.5300

By difficulty:
{
  "high_distractor": {
    "count": 500.0,
    "accuracy": 0.53
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.53
  }
}
Saved predictions to: new_data2/outputs/bert_mem/high_distractor/test_predictions.json

### Discussion


## 2. Attention Biasing

Another possible cause of retrieval failure is not only insufficient aggregation capacity, but also poor information allocation. In long-context settings with many distractors, self-attention may distribute attention weights too broadly across irrelevant tokens, causing sparse task-relevant signals to receive insufficient focus.

To test whether guided attention allocation improves selective retrieval, we introduce an attention bias mechanism that modifies the attention scores before the softmax operation. The goal is not to explicitly reveal the correct answer to the model, but rather to encourage stronger attention toward potentially informative regions of the context.

Instead of using the standard attention formulation
$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V,
$$

we modify the attention logits by adding a bias term:
$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d}} + B
\right)V,
$$
where $B$ denotes a bias matrix that adjusts the relative attention preference between tokens.

### Theoretical Intuition

In standard self-attention, attention weights are determined entirely by token similarity:
$$
\alpha_{ij}
=
\frac{
\exp(e_{ij})
}{
\sum_k \exp(e_{ik})
},
$$
where
$$
e_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d}}.
$$

Under long-horizon sparse retrieval settings, many irrelevant distractor tokens may produce competing attention scores, causing the attention distribution to become diluted across the sequence.

By introducing an additional bias term,
$$
e_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d}}
+
B_{ij},
$$
the model can preferentially allocate attention toward selected regions or token types.

In our experiments, the bias mechanism serves as a controlled intervention for studying whether selective attention allocation improves long-range retrieval robustness under distractor-heavy conditions.

Importantly, the purpose of this modification is not to engineer a task-specific heuristic, but rather to test whether retrieval failures are partially caused by insufficient attention concentration on sparse task-relevant information.

### Result

#### MultiWOZ

#### Synthetic

Loading data...
Rendering rows...
Model: bert_attn_bias
Device: mps
Max length: 512
Train rows: 4000 | Val rows: 2000 | Test rows: 3564
Train batches: 500 | Val batches: 125 | Test batches: 223
Train batch size: 8 | Eval batch size: 16

Training setup | Epochs: 3 | Train batches/epoch: 500 | Val batches: 125 | Total steps: 1500 | Warmup steps: 150 | Device: mps

Starting epoch 1/3 (500 train batches)
Epoch 1/3 | Batch 1/500 | Loss: 0.7816 | Avg Loss: 0.7816 | LR: 1.33e-07
Epoch 1/3 | Batch 50/500 | Loss: 0.4614 | Avg Loss: 0.6275 | LR: 6.67e-06
Epoch 1/3 | Batch 100/500 | Loss: 0.4403 | Avg Loss: 0.5838 | LR: 1.33e-05
Epoch 1/3 | Batch 150/500 | Loss: 0.5186 | Avg Loss: 0.5910 | LR: 2.00e-05
Epoch 1/3 | Batch 200/500 | Loss: 0.5669 | Avg Loss: 0.5866 | LR: 1.93e-05
Epoch 1/3 | Batch 250/500 | Loss: 0.3337 | Avg Loss: 0.5742 | LR: 1.85e-05
Epoch 1/3 | Batch 300/500 | Loss: 0.4406 | Avg Loss: 0.5697 | LR: 1.78e-05
Epoch 1/3 | Batch 350/500 | Loss: 0.5952 | Avg Loss: 0.5677 | LR: 1.70e-05
Epoch 1/3 | Batch 400/500 | Loss: 0.8787 | Avg Loss: 0.5635 | LR: 1.63e-05
Epoch 1/3 | Batch 450/500 | Loss: 0.3624 | Avg Loss: 0.5623 | LR: 1.56e-05
Epoch 1/3 | Batch 500/500 | Loss: 0.3640 | Avg Loss: 0.5569 | LR: 1.48e-05
Evaluating epoch 1/3...
Epoch 1/3 | Train Loss: 0.5569 | Val Sample Acc: 0.5780
Saved new best model to: new_data2/outputs/bert_attn_bias/best_model.pt

Starting epoch 2/3 (500 train batches)
Epoch 2/3 | Batch 1/500 | Loss: 0.5438 | Avg Loss: 0.5438 | LR: 1.48e-05
Epoch 2/3 | Batch 50/500 | Loss: 0.1911 | Avg Loss: 0.5347 | LR: 1.41e-05
Epoch 2/3 | Batch 100/500 | Loss: 0.5066 | Avg Loss: 0.5161 | LR: 1.33e-05
Epoch 2/3 | Batch 150/500 | Loss: 1.0143 | Avg Loss: 0.5213 | LR: 1.26e-05
Epoch 2/3 | Batch 200/500 | Loss: 0.3832 | Avg Loss: 0.5100 | LR: 1.19e-05
Epoch 2/3 | Batch 250/500 | Loss: 0.3012 | Avg Loss: 0.5052 | LR: 1.11e-05
Epoch 2/3 | Batch 300/500 | Loss: 0.2252 | Avg Loss: 0.5028 | LR: 1.04e-05
Epoch 2/3 | Batch 350/500 | Loss: 0.3142 | Avg Loss: 0.5023 | LR: 9.63e-06
Epoch 2/3 | Batch 400/500 | Loss: 0.1743 | Avg Loss: 0.5017 | LR: 8.89e-06
Epoch 2/3 | Batch 450/500 | Loss: 0.2789 | Avg Loss: 0.5066 | LR: 8.15e-06
Epoch 2/3 | Batch 500/500 | Loss: 0.3805 | Avg Loss: 0.5047 | LR: 7.41e-06
Evaluating epoch 2/3...
Epoch 2/3 | Train Loss: 0.5047 | Val Sample Acc: 0.6340
Saved new best model to: new_data2/outputs/bert_attn_bias/best_model.pt

Starting epoch 3/3 (500 train batches)
Epoch 3/3 | Batch 1/500 | Loss: 0.3033 | Avg Loss: 0.3033 | LR: 7.39e-06
Epoch 3/3 | Batch 50/500 | Loss: 0.4254 | Avg Loss: 0.5093 | LR: 6.67e-06
Epoch 3/3 | Batch 100/500 | Loss: 0.2017 | Avg Loss: 0.4638 | LR: 5.93e-06
Epoch 3/3 | Batch 150/500 | Loss: 0.6154 | Avg Loss: 0.4723 | LR: 5.19e-06
Epoch 3/3 | Batch 200/500 | Loss: 0.1958 | Avg Loss: 0.4658 | LR: 4.44e-06
Epoch 3/3 | Batch 250/500 | Loss: 0.3793 | Avg Loss: 0.4670 | LR: 3.70e-06
Epoch 3/3 | Batch 300/500 | Loss: 1.0185 | Avg Loss: 0.4675 | LR: 2.96e-06
Epoch 3/3 | Batch 350/500 | Loss: 0.2956 | Avg Loss: 0.4589 | LR: 2.22e-06
Epoch 3/3 | Batch 400/500 | Loss: 0.3047 | Avg Loss: 0.4604 | LR: 1.48e-06
Epoch 3/3 | Batch 450/500 | Loss: 1.0487 | Avg Loss: 0.4636 | LR: 7.41e-07
Epoch 3/3 | Batch 500/500 | Loss: 0.2199 | Avg Loss: 0.4587 | LR: 0.00e+00
Evaluating epoch 3/3...
Epoch 3/3 | Train Loss: 0.4587 | Val Sample Acc: 0.6720
Saved new best model to: new_data2/outputs/bert_attn_bias/best_model.pt

===== Validation =====
Sample-level accuracy: 0.6720

By difficulty:
{
  "hard": {
    "count": 500.0,
    "accuracy": 0.672
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.672
  }
}

===== Test =====
Sample-level accuracy: 0.6510

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6509539842873177
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6509539842873177
  }
}

Saved results to: new_data2/outputs/bert_attn_bias

---
### Full Context
===== New Test Set =====
Sample-level accuracy: 0.6532

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6531986531986532
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6531986531986532
  }
}
Saved predictions to: new_data2/outputs/bert_attn_bias/test_full/test_predictions.json

---
### Remove Signal
===== New Test Set =====
Sample-level accuracy: 0.2065

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.20650953984287318
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.20650953984287318
  }
}
Saved predictions to: new_data2/outputs/bert_attn_bias/test_rm_signal/test_predictions.json

---

### Local Only (4 nearest sentence)
===== New Test Set =====
Sample-level accuracy: 0.3064

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.3063973063973064
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.3063973063973064
  }
}
Saved predictions to: new_data2/outputs/bert_attn_bias/test_local/test_predictions.json

---
### Candidate Only
===== New Test Set =====
Sample-level accuracy: 0.2738

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2738496071829405
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2738496071829405
  }
}
Saved predictions to: new_data2/outputs/bert_attn_bias/test_candidate/test_predictions.json

---

### Short Distance
===== New Test Set =====
Sample-level accuracy: 0.6380

By difficulty:
{
  "short_distance": {
    "count": 500.0,
    "accuracy": 0.638
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.638
  }
}
Saved predictions to: new_data2/outputs/bert_attn_bias/test_short_distance/test_predictions.json


#### Long Distance
===== New Test Set =====
Sample-level accuracy: 0.5960

By difficulty:
{
  "long_distance": {
    "count": 500.0,
    "accuracy": 0.596
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.596
  }
}
Saved predictions to: new_data2/outputs/bert_attn_bias/test_long_distance/test_predictions.json

#### High Distractor
===== New Test Set =====
Sample-level accuracy: 0.5120

By difficulty:
{
  "high_distractor": {
    "count": 500.0,
    "accuracy": 0.512
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.512
  }
}
Saved predictions to: new_data2/outputs/bert_attn_bias/high_distractor/test_predictions.json


## Bert
Loading data...
Rendering rows...
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Model: bert
Device: mps
Max length: 512
Train rows: 4000 | Val rows: 2000 | Test rows: 3564
Train batches: 500 | Val batches: 125 | Test batches: 223
Train batch size: 8 | Eval batch size: 16

Training setup | Epochs: 3 | Train batches/epoch: 500 | Val batches: 125 | Total steps: 1500 | Warmup steps: 150 | Device: mps

Starting epoch 1/3 (500 train batches)
Epoch 1/3 | Batch 1/500 | Loss: 0.7602 | Avg Loss: 0.7602 | LR: 1.33e-07
Epoch 1/3 | Batch 50/500 | Loss: 0.3334 | Avg Loss: 0.6919 | LR: 6.67e-06
Epoch 1/3 | Batch 100/500 | Loss: 0.7622 | Avg Loss: 0.6289 | LR: 1.33e-05
Epoch 1/3 | Batch 150/500 | Loss: 0.9543 | Avg Loss: 0.6018 | LR: 2.00e-05
Epoch 1/3 | Batch 200/500 | Loss: 0.2478 | Avg Loss: 0.6029 | LR: 1.93e-05
Epoch 1/3 | Batch 250/500 | Loss: 0.6829 | Avg Loss: 0.5975 | LR: 1.85e-05
Epoch 1/3 | Batch 300/500 | Loss: 0.3818 | Avg Loss: 0.5892 | LR: 1.78e-05
Epoch 1/3 | Batch 350/500 | Loss: 0.7528 | Avg Loss: 0.5887 | LR: 1.70e-05
Epoch 1/3 | Batch 400/500 | Loss: 0.7968 | Avg Loss: 0.5778 | LR: 1.63e-05
Epoch 1/3 | Batch 450/500 | Loss: 0.1823 | Avg Loss: 0.5750 | LR: 1.56e-05
Epoch 1/3 | Batch 500/500 | Loss: 0.4756 | Avg Loss: 0.5787 | LR: 1.48e-05
Evaluating epoch 1/3...
Epoch 1/3 | Train Loss: 0.5787 | Val Sample Acc: 0.5560
Saved new best model to: new_data2/outputs/bert/best_model.pt

Starting epoch 2/3 (500 train batches)
Epoch 2/3 | Batch 1/500 | Loss: 0.6231 | Avg Loss: 0.6231 | LR: 1.48e-05
Epoch 2/3 | Batch 50/500 | Loss: 0.7718 | Avg Loss: 0.5486 | LR: 1.41e-05
Epoch 2/3 | Batch 100/500 | Loss: 0.3236 | Avg Loss: 0.5171 | LR: 1.33e-05
Epoch 2/3 | Batch 150/500 | Loss: 0.5635 | Avg Loss: 0.5201 | LR: 1.26e-05
Epoch 2/3 | Batch 200/500 | Loss: 0.6119 | Avg Loss: 0.5261 | LR: 1.19e-05
Epoch 2/3 | Batch 250/500 | Loss: 0.6036 | Avg Loss: 0.5264 | LR: 1.11e-05
Epoch 2/3 | Batch 300/500 | Loss: 0.7336 | Avg Loss: 0.5291 | LR: 1.04e-05
Epoch 2/3 | Batch 350/500 | Loss: 0.7138 | Avg Loss: 0.5283 | LR: 9.63e-06
Epoch 2/3 | Batch 400/500 | Loss: 0.9509 | Avg Loss: 0.5296 | LR: 8.89e-06
Epoch 2/3 | Batch 450/500 | Loss: 0.4709 | Avg Loss: 0.5242 | LR: 8.15e-06
Epoch 2/3 | Batch 500/500 | Loss: 0.1280 | Avg Loss: 0.5219 | LR: 7.41e-06
Evaluating epoch 2/3...
Epoch 2/3 | Train Loss: 0.5219 | Val Sample Acc: 0.5620
Saved new best model to: new_data2/outputs/bert/best_model.pt

Starting epoch 3/3 (500 train batches)
Epoch 3/3 | Batch 1/500 | Loss: 0.4041 | Avg Loss: 0.4041 | LR: 7.39e-06
Epoch 3/3 | Batch 50/500 | Loss: 0.5050 | Avg Loss: 0.4901 | LR: 6.67e-06
Epoch 3/3 | Batch 100/500 | Loss: 0.5155 | Avg Loss: 0.4858 | LR: 5.93e-06
Epoch 3/3 | Batch 150/500 | Loss: 0.5624 | Avg Loss: 0.4932 | LR: 5.19e-06
Epoch 3/3 | Batch 200/500 | Loss: 0.4437 | Avg Loss: 0.4841 | LR: 4.44e-06
Epoch 3/3 | Batch 250/500 | Loss: 0.2644 | Avg Loss: 0.4844 | LR: 3.70e-06
Epoch 3/3 | Batch 300/500 | Loss: 0.6837 | Avg Loss: 0.4846 | LR: 2.96e-06
Epoch 3/3 | Batch 350/500 | Loss: 0.2737 | Avg Loss: 0.4817 | LR: 2.22e-06
Epoch 3/3 | Batch 400/500 | Loss: 0.2661 | Avg Loss: 0.4756 | LR: 1.48e-06
Epoch 3/3 | Batch 450/500 | Loss: 0.6655 | Avg Loss: 0.4788 | LR: 7.41e-07
Epoch 3/3 | Batch 500/500 | Loss: 0.2837 | Avg Loss: 0.4763 | LR: 0.00e+00
Evaluating epoch 3/3...
Epoch 3/3 | Train Loss: 0.4763 | Val Sample Acc: 0.5860
Saved new best model to: new_data2/outputs/bert/best_model.pt

===== Validation =====
Sample-level accuracy: 0.5860

By difficulty:
{
  "hard": {
    "count": 500.0,
    "accuracy": 0.586
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.586
  }
}

===== Test =====
Sample-level accuracy: 0.6296

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6296296296296297
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6296296296296297
  }
}

Saved results to: new_data2/outputs/bert

---
### Full Context
===== New Test Set =====
Sample-level accuracy: 0.6498

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6498316498316499
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.6498316498316499
  }
}
Saved predictions to: new_data2/outputs/bert/test_full/test_predictions.json

---
### Remove Signal
===== New Test Set =====
Sample-level accuracy: 0.2043

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.20426487093153758
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.20426487093153758
  }
}
Saved predictions to: new_data2/outputs/bert/test_rm_signal/test_predictions.json

---
### Local Only (4 nearest sentence)
===== New Test Set =====
Sample-level accuracy: 0.2828

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2828282828282828
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2828282828282828
  }
}
Saved predictions to: new_data2/outputs/bert/test_local/test_predictions.json

---
### Candidate Only
===== New Test Set =====
Sample-level accuracy: 0.2559

By difficulty:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2558922558922559
  }
}

By has_distractor:
{
  "missing": {
    "count": 891.0,
    "accuracy": 0.2558922558922559
  }
}
Saved predictions to: new_data2/outputs/bert/test_candidate/test_predictions.json

---

### Short Distance
===== New Test Set =====
Sample-level accuracy: 0.5440

By difficulty:
{
  "short_distance": {
    "count": 500.0,
    "accuracy": 0.544
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.544
  }
}
Saved predictions to: new_data2/outputs/bert/test_short_distance/test_predictions.json

---
### Long Distance
===== New Test Set =====
Sample-level accuracy: 0.5880

By difficulty:
{
  "long_distance": {
    "count": 500.0,
    "accuracy": 0.588
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.588
  }
}
Saved predictions to: new_data2/outputs/bert/test_long_distance/test_predictions.json

---

### Hight Distractor
===== New Test Set =====
Sample-level accuracy: 0.5000

By difficulty:
{
  "high_distractor": {
    "count": 500.0,
    "accuracy": 0.5
  }
}

By has_distractor:
{
  "True": {
    "count": 500.0,
    "accuracy": 0.5
  }
}
Saved predictions to: new_data2/outputs/bert/high_distractor/test_predictions.json