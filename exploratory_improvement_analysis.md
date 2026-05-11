# Exploratory Improvement Analysis

This document summarizes the exploratory BERT variants used to diagnose retrieval failures under sparse long-context settings. The goal is not to claim a new state-of-the-art model, but to test which failure modes are most plausible in the controlled benchmark.

## Motivation

The baseline BERT ranker must compress the entire rendered dialogue and candidate into a single classification representation. In sparse retrieval settings, the task-relevant signal may appear far from the query and may be surrounded by distractors. We therefore evaluate two interventions:

1. **Memory-enhanced BERT (`bert_mem`)**: adds explicit `[MEM]` tokens as additional global aggregation slots.
2. **Attention-biased BERT (`bert_attn_bias`)**: adds a positive attention-logit bias toward context tokens that overlap with the candidate.

These variants probe two related hypotheses:

- Retrieval failures may come from insufficient global aggregation capacity.
- Retrieval failures may come from poor attention allocation toward sparse informative regions.

## Model Variants

| Model | Main change | Max length used | Interpretation |
|---|---:|---:|---|
| `bert` | Standard BERT sequence classifier | 512 | Baseline ranking model |
| `bert_mem` | Prepends 4 `[MEM]` tokens and classifies with `[CLS] + mean([MEM])` | 512 | Tests whether extra aggregation slots help |
| `bert_attn_bias` | Adds candidate-overlap attention bias before softmax | 512 | Tests whether guided attention allocation helps |

All three BERT-based variants are evaluated with `max_length=512`, so the comparison is not confounded by different context-window sizes.

## Mathematical Formulation

### Baseline BERT Ranker

Each ranking example is represented as a sequence pair:

$$
X = [\texttt{[CLS]}, c_1, \dots, c_m, \texttt{[SEP]}, a_1, \dots, a_n, \texttt{[SEP]}],
$$

where \(c_i\) are dialogue-context tokens and \(a_i\) are candidate tokens. BERT produces contextual states:

$$
H = \mathrm{BERT}(X),
$$

and the standard classifier uses only the final `[CLS]` representation:

$$
z = h_{\texttt{[CLS]}},
\qquad
\hat{y} = \mathrm{softmax}(Wz + b).
$$

This means all evidence from the dialogue and candidate must be compressed into a single vector before classification.

### Memory-Enhanced BERT

For `bert_mem`, we prepend \(k\) memory tokens to the context:

$$
\tilde{X}
=
[\texttt{[CLS]}, m_1, \dots, m_k, c_1, \dots, c_m, \texttt{[SEP]}, a_1, \dots, a_n, \texttt{[SEP]}].
$$

Because the memory tokens participate in normal self-attention, each memory state can aggregate information from the full visible input:

$$
h_{m_i}^{(\ell+1)}
=
\sum_{j=1}^{|\tilde{X}|}
\alpha_{ij}^{(\ell)} W_V h_j^{(\ell)},
$$

where

$$
\alpha_{ij}^{(\ell)}
=
\mathrm{softmax}_j
\left(
\frac{
(W_Q h_i^{(\ell)})^\top (W_K h_j^{(\ell)})
}{
\sqrt{d}
}
\right).
$$

Instead of classifying from `[CLS]` alone, the implementation averages the memory-token states and concatenates them with `[CLS]`:

$$
\bar{m}
=
\frac{1}{k}\sum_{i=1}^k h_{m_i},
\qquad
z
=
[h_{\texttt{[CLS]}} ; \bar{m}],
$$

then predicts:

$$
\hat{y}
=
\mathrm{softmax}(Wz + b).
$$

In the current implementation, \(k=4\). This tests whether additional trainable aggregation slots improve sparse evidence retrieval.

### Attention-Biased BERT

Standard self-attention computes attention logits:

$$
e_{ij}
=
\frac{
q_i^\top k_j
}{
\sqrt{d}
}.
$$

The attention-biased variant adds a bias term before the softmax:

$$
e_{ij}
=
\frac{
q_i^\top k_j
}{
\sqrt{d}
}
+
B_{ij}.
$$

Then attention weights become:

$$
\alpha_{ij}
=
\frac{
\exp(e_{ij})
}{
\sum_t \exp(e_{it})
}.
$$

In the implementation, the bias is candidate-conditioned and token-position based. Let \(S(a)\) be the set of non-stopword candidate token ids. For a context token position \(j\):

$$
b_j
=
\begin{cases}
1, & x_j \in S(a) \text{ and } x_j \text{ is a context token}, \\
0, & \text{otherwise}.
\end{cases}
$$

This vector is broadcast across query positions and attention heads:

$$
B_{ij}
=
\lambda b_j,
$$

where \(\lambda\) is the bias strength. The current implementation uses \(\lambda = 1.0\). This does not directly reveal the correct answer, but it encourages attention toward context tokens that lexically overlap with the candidate.

## Evaluation Sets

| Set | Samples | Purpose |
|---|---:|---|
| Validation | 500 | In-domain validation set |
| Original Test | 891 | Standard held-out test set |
| Full Context | 891 | Full rendered dialogue context |
| Remove Signal | 891 | Removes all turns with `role == "signal"` |
| Local Only | 891 | Keeps only local context near query |
| Candidate Only | 891 | Removes dialogue context, tests candidate prior |
| Short Distance | 500 | Signal close to query |
| Long Distance | 500 | Signal farther from query |
| High Distractor | 500 | Long distance plus more distractors/noise |

For the generated distance sets, `signal_query_distance` is measured in dialogue turns, not tokens.

| Set | Turn-distance range | Average distance | Distractors | Noise blocks |
|---|---:|---:|---:|---:|
| Short Distance | 1-6 | 3.17 | 3 | 5 |
| Long Distance | 21-36 | 28.07 | 4 | 30 |
| High Distractor | 21-40 | 30.23 | 8 | 24 |

The average rendered turn length is approximately 10 BERT tokens, so a 28-turn distance is roughly 280 tokens before accounting for candidate tokens and special tokens.

## Main Results

Sample-level accuracy:

| Evaluation | `bert` | `bert_mem` | `bert_attn_bias` |
|---|---:|---:|---:|
| Validation | 0.5860 | 0.6440 | **0.6720** |
| Original Test | 0.6296 | **0.7003** | 0.6510 |
| Full Context | 0.6498 | **0.7104** | 0.6532 |
| Remove Signal | 0.2043 | **0.2334** | 0.2065 |
| Local Only | 0.2828 | 0.2963 | **0.3064** |
| Candidate Only | 0.2559 | 0.2312 | **0.2738** |
| Short Distance | 0.5440 | **0.6400** | 0.6380 |
| Long Distance | 0.5880 | 0.5880 | **0.5960** |
| High Distractor | 0.5000 | **0.5300** | 0.5120 |

Random baseline is approximately 0.25 because each sample has four candidates.

## Analysis

### 1. Full-context performance improves with memory tokens

`bert_mem` improves over baseline BERT on the main test settings:

| Setting | BERT | MemBERT | Difference |
|---|---:|---:|---:|
| Original Test | 0.6296 | 0.7003 | +0.0707 |
| Full Context | 0.6498 | 0.7104 | +0.0606 |
| Short Distance | 0.5440 | 0.6400 | +0.0960 |

This supports the idea that extra aggregation slots can help the model preserve sparse task-relevant information beyond the standard `[CLS]` representation.

### 2. Removing signal collapses performance near random

All models drop close to the four-way random baseline when signal turns are removed:

| Model | Remove Signal |
|---|---:|
| `bert` | 0.2043 |
| `bert_mem` | 0.2334 |
| `bert_attn_bias` | 0.2065 |

This suggests the models are not solving the task from candidate priors alone. The signal turns are necessary. The accuracy is not near zero because removing signal does not force the model to consistently choose a particular false candidate; it mostly produces noisy guessing.

### 3. Candidate-only and local-only settings are weak

Candidate-only accuracy stays around random:

| Model | Candidate Only |
|---|---:|
| `bert` | 0.2559 |
| `bert_mem` | 0.2312 |
| `bert_attn_bias` | 0.2738 |

Local-only is only slightly better:

| Model | Local Only |
|---|---:|
| `bert` | 0.2828 |
| `bert_mem` | 0.2963 |
| `bert_attn_bias` | 0.3064 |

This supports the benchmark assumption that the answer usually depends on non-local evidence rather than the candidate text alone or the immediately preceding turns.

### 4. Attention bias helps short-distance and validation performance, but is weaker under stress

After rerunning `bert_attn_bias` with `max_length=512`, the attention-bias variant no longer shows the large stress-test gains seen in the earlier 256-token run. Its strongest results are validation, local-only, candidate-only, and short-distance performance:

| Setting | BERT | MemBERT | Attention Bias |
|---|---:|---:|---:|
| Validation | 0.5860 | 0.6440 | **0.6720** |
| Local Only | 0.2828 | 0.2963 | **0.3064** |
| Candidate Only | 0.2559 | 0.2312 | **0.2738** |
| Short Distance | 0.5440 | **0.6400** | 0.6380 |

On the harder generated stress sets, memory tokens are more reliable:

| Setting | BERT | MemBERT | Attention Bias |
|---|---:|---:|---:|
| Long Distance | 0.5880 | 0.5880 | **0.5960** |
| High Distractor | 0.5000 | **0.5300** | 0.5120 |

This suggests that the current attention-bias heuristic helps when the relevant lexical overlap is nearby or easy to exploit, but it does not robustly solve the harder long-distance or high-distractor settings.

However, this result should be interpreted carefully because the current attention-bias implementation uses lexical candidate-context overlap. It is a controlled intervention, but it is still a heuristic signal. It does not reveal the correct answer directly, but it may be especially well matched to this synthetic slot-value dataset.

### 5. Current long-distance setting is moderate, not extreme

The current long-distance set has an average turn distance of 28.07. Since one turn is roughly 10 BERT tokens, this corresponds to about 280 tokens. With `max_length=512`, the signal can often still be inside the model window. This is useful for testing retrieval under distance, rather than pure truncation failure.

If the distance were increased to hundreds of turns, BERT would likely never see the signal. That would test context-window truncation rather than attention or aggregation failure.

## Conclusions

The results support three main conclusions:

1. The task depends on the sparse signal: removing signal drops performance close to random.
2. Memory tokens improve the main full-context setting, suggesting that additional aggregation capacity helps.
3. Attention biasing helps in some easier or more lexical-overlap-driven settings, but it is less robust than memory tokens on full-context and high-distractor evaluation.

