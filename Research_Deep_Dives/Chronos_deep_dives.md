# Chronos: Learning the Language of Time Series
**Ansari et al., 2024 | AWS AI Labs, UC San Diego, University of Freiburg**  
arXiv: 2403.07815v3

---

## What this paper is about

Chronos reframes time series forecasting as a language modeling problem. Instead of building yet another architecture with time-series-specific components, the authors ask whether a standard language model — with minimal changes — can forecast numerical sequences. The answer is largely yes.

The core observation is that language models and forecasting models are doing structurally similar things: both model sequential data to predict what comes next. The gap is representational. Language models operate over a finite token vocabulary; time series values are continuous and unbounded. Chronos bridges this by converting real values into discrete tokens via scaling and quantization, then training an off-the-shelf transformer on those tokens using cross-entropy loss.

---

## Method

**Tokenization pipeline:**

1. *Scaling* — each time series is normalized by the mean of absolute values in the context window (mean scaling). This maps different-magnitude series into a comparable range without destroying zero values (e.g., zero sales, zero solar output at night).

2. *Quantization* — the scaled values are mapped to one of B discrete bins. The authors use uniform binning over the interval [−15, +15], with B = 4094 bins plus two special tokens (PAD, EOS). Uniform bins were chosen over data-dependent quantile bins because downstream datasets can have very different distributions from training data.

3. *Language model training* — the token sequences are fed into a T5 encoder-decoder model trained with standard cross-entropy loss. No architectural modifications are needed beyond resizing the embedding layers to match the new vocabulary size.

At inference, the model autoregressively samples token sequences, which are then dequantized and unscaled to produce probabilistic forecast trajectories.

**What this does not use:** time features (day-of-week, seasonality flags), lag features, frequency information. The model treats the series as a plain sequence.

---

## Data

Public time series data is scarce compared to NLP. The authors address this with two augmentation strategies:

**TSMixup** — randomly samples k time series from different datasets, scales them, and takes a convex combination weighted by a Dirichlet distribution. This creates new time series with mixed patterns, improving distribution coverage without requiring new data collection.

**KernelSynth** — generates synthetic time series using Gaussian processes. A kernel bank contains basis kernels (linear, RBF, periodic, rational quadratic, constant, white noise). For each synthetic series, j kernels are sampled and combined via random binary operations (+ or ×). The resulting composite kernel defines a GP prior from which a time series is drawn.

Training corpus: 28 datasets (~890K univariate series, ~84B tokens), augmented with 10M TSMixup samples and 1M KernelSynth synthetic series. Sampling ratio during training: 9:1 (real augmentations to synthetic).

---

## Model variants

| Model | Params | Architecture |
|---|---|---|
| Chronos-T5 Mini | 20M | T5 encoder-decoder |
| Chronos-T5 Small | 46M | T5 encoder-decoder |
| Chronos-T5 Base | 200M | T5 encoder-decoder |
| Chronos-T5 Large | 710M | T5 encoder-decoder |
| Chronos-GPT2 | 90M | Decoder-only |

All trained with AdamW, lr=0.001 annealed to 0 over 200K steps, batch size 256, on 8× A100 (40GB) GPUs. Context length: 512 tokens. Prediction length during training: 64 (covers all evaluation tasks).

---

## Evaluation

**Benchmarks:**
- Benchmark I (15 datasets): used for both training and in-domain evaluation
- Benchmark II (27 datasets): zero-shot evaluation only — Chronos never sees these during training

**Metrics:**
- WQL (Weighted Quantile Loss) for probabilistic forecasting — evaluated at 9 quantile levels {0.1, ..., 0.9}
- MASE (Mean Absolute Scaled Error) for point forecasting — uses median forecast

Scores are normalized relative to Seasonal Naive, then aggregated via geometric mean across datasets (arithmetic mean would be misleading for relative scores).

**Baselines include:** AutoETS, AutoARIMA, AutoTheta, SCUM ensemble, DeepAR, TFT, PatchTST, DLinear, WaveNet, N-BEATS, N-HiTS, GPT4TS, Lag-Llama, Moirai-1.0-R, LLMTime, ForecastPFN.

---

## Results

**In-domain (Benchmark I):**  
Chronos-T5 Base and Large outperform all baselines — including task-specific deep learning models trained on each dataset individually. Chronos-T5 Mini (20M) beats Moirai-1.0-R Large (311M), which was trained on a substantially larger corpus.

**Zero-shot (Benchmark II):**  
Chronos models significantly outperform statistical baselines and other pretrained models (Lag-Llama, LLMTime, ForecastPFN). They perform comparably to the best task-specific deep learning models that were explicitly trained on these datasets. Chronos-T5 Large ranks 2nd on point forecasting, above the SCUM ensemble.

**Fine-tuning:**  
Chronos-T5 Small fine-tuned on individual Benchmark II datasets (1000 steps, lr=0.001) takes the top spot overall on Benchmark II, surpassing both larger zero-shot Chronos variants and task-specific baselines. This uses very little compute relative to training from scratch.

---

## Ablations and design choices

**Model size** — larger models consistently improve both in-domain and zero-shot performance. The scaling trend holds across all four sizes tested.

**Initialization** — randomly initialized models converge to lower training loss than models initialized from pretrained T5 language model weights. Language pretraining offers no consistent benefit for time series forecasting. Larger language-model-initialized models converge faster initially but plateau higher than random init.

**TSMixup** — similar in-domain performance with or without it, but zero-shot performance improves with TSMixup. The diversity introduced by mixing series from different datasets helps generalization.

**Synthetic data proportion** — best results at ~10% KernelSynth data in the training mix. More synthetic data degrades performance. A model trained on 100% synthetic data still performs reasonably well in absolute terms, better than ForecastPFN (which also trains only on synthetic data).

**Context length** — performance improves up to 1024 tokens, then saturates or degrades. The evaluation benchmark is likely undersampling high-frequency series where longer context would matter more.

**Vocabulary size** — larger vocabulary improves MASE (scale-invariant) monotonically. WQL (scale-dependent) initially improves then degrades. The trade-off: too few bins means coarse discretization error; too many bins means sparse coverage per bin, harming generalization.

---

## Qualitative behavior

- **Linear trend**: forecasts correctly
- **Exponential trend**: struggles — likely underrepresented in training data
- **Short context**: underestimates trend magnitude; longer context corrects this (demonstrated on the Air Passengers dataset)
- **Seasonality**: accurate, even for complex multi-period seasonality
- **Additive/multiplicative combinations**: handled well
- **AR processes**: competitive with fitted AR models, especially as process order increases. AutoARIMA is outperformed on AR(3) and AR(4).
- **Predictive distributions**: despite cross-entropy being distance-agnostic, the model learns to place probability mass on neighboring bins. Multimodal distributions are produced when data supports them.

**Known failure modes:**  
- Sparse series with occasional large spikes: the mean scale becomes very small, so large spike values fall outside the representable range [−15s, +15s]
- Series with large mean relative to variance: tokens are spaced too far apart, losing precision in the signal variation

---

## Key arguments made

- Time-series-specific architectural components (lags, time features, patching) are not necessary for competitive forecasting performance.
- A categorical output distribution (regression via classification) is flexible enough to model arbitrary predictive distributions, including multimodal ones, without constraining the model to a parametric family.
- Pretrained language model weights are not useful for time series. Random initialization is preferable or equivalent.
- The main bottleneck for this approach is data quality and quantity, not modeling — echoing what NLP learned a decade earlier.

---

## Limitations

- Univariate only; no native support for covariates or multivariate dependencies
- Fixed tokenization range can fail on sparse or high-mean series
- Inference is slower than task-specific models for the larger variants
- Exponential trends are poorly handled without preprocessing
- The benchmark has limited high-frequency data, so the impact of longer context lengths is hard to assess conclusively

---

## Practical notes

- Models and weights are publicly released
- Fine-tuning can be done with minimal compute (1000 steps), and LoRA or conformal calibration are suggested as further options
- Inference can be accelerated with standard LLM techniques (quantization, speculative decoding, FlashAttention)
- Training cost for the Large model: ~63 GPU-hours on a p4d.24xlarge instance (~$2,066 at on-demand pricing)
