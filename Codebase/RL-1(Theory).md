# Paper 01: Diversified Scaling Inference in Time Series Foundation Models
**Title:** Diversified Scaling Inference in TSFMs (Jan 2026)

### Overview
The advancement of Time Series Foundation Models (TSFMs) has been primarily driven by large-scale pre-training. However, this research demonstrates that inference-time compute—the computational resources expended at test time—remains an untapped lever for performance. Drawing a parallel to "Chain of Thought" or "Self-Consistency" in Large Language Models (e.g., OpenAI o1), this paper proves that a fixed TSFM can achieve significantly higher accuracy without additional training by utilizing Diversified Scaling Inference (DSI). By generating multiple diversified trajectories through input-level perturbations and aggregating them, the model can navigate the probability distribution to find the ground truth more effectively.

---

### The Experimental Environment

#### 1. The Models (The Brains)
*   **Chronos (Amazon):** A language-centric model that tokenizes time series values into discrete "bins" using the T5 (Encoder-Decoder) architecture.
*   **TimesFM (Google):** A decoder-only (GPT-style) model that utilizes "patching" to group time points into segments for long-term pattern recognition.
*   **Moirai (Salesforce):** A universal forecasting model designed to handle any data frequency (hourly, daily, weekly) within a single architecture.
*   **Time-MoE (Microsoft):** A Mixture-of-Experts model that routes different temporal patterns to specialized sub-networks to manage diverse data "vibes."

#### 2. The Datasets (The Testing Ground)
*   **ETTh1:** Electricity Transformer Temperature data; the benchmark standard for assessing time series model precision.
*   **Traffic:** Highway occupancy metrics; characterized by high noise and complexity, making it ideal for testing inference scaling limits.
*   **Electricity:** Large-scale consumption patterns; used to evaluate the model's ability to handle strong seasonality.
*   **Weather:** Temperature and pressure logs; used to test the stability of cyclical predictions.

---

### Core Technical Pillars

#### 1. Pre-training vs. Inference Scaling
Traditional scaling laws focus on increasing model parameters and dataset size during training. DSI introduces a new dimension: trading test-time compute for accuracy. The results confirm that a smaller model (Chronos-Tiny) can outperform a larger model (Chronos-Large) simply by allocating more samples ($N$) at inference time.

#### 2. Diversity-Fidelity Trade-off
Successful scaling requires a balance between expanding the search space (Diversity) and maintaining the original signal (Fidelity).
*   **Support Expansion:** DSI "shakes" the input data to allow the model to explore regions of the distribution that are otherwise hidden by the model's fixed training bias.
*   **Relevance Band:** The research finds that perturbations work best when the cosine similarity between the original and "shaken" data remains between 0.95 and 1.00.

#### 3. Perturbation Taxonomy (Algorithm 1)
To induce beneficial diversity, the paper introduces **Algorithm 1 (Structure-Sensitive Perturbation)**. Unlike standard Gaussian noise, this method uses STL Decomposition to split the series into Trend, Seasonality, and Residual components. It specifically targets "unstable" regions (high-gradient segments) to apply higher-intensity perturbations, focusing the model's "thinking" on the most difficult parts of the sequence.

---

### Mathematical Innovations

#### 1. The Critical Sample Threshold ($N^*$)
Introducing diversity is a "risk-reward" calculation. If the sample budget ($N$) is too low, the noise may degrade performance. DSI only becomes statistically superior to standard sampling once $N$ exceeds the threshold $N^*$.

**Formula:**
$$N^* = \frac{\ln\left(\frac{L_{bad}-L_{good}}{L_0-L_{good}}\right)}{\ln\left(\frac{1}{1-\rho}\right)}$$

*   **Interpretation:** At high $N$, the probability ($\rho$) of hitting a superior "good" sample outweighs the risk of bad samples, making DSI the mathematically superior strategy for high-compute environments.

#### 2. RobustMSE
Standard MSE evaluates a single point-estimate. RobustMSE is proposed to quantify the "headroom" of a model by measuring repeatable performance across multiple diversified trials under a fixed budget.

$$
RobustMSE_{EM|MV} = \frac{1}{T} \sum_{i=1}^{T} MSE(\hat{Y}^i_{EM|MV}, Y)
$$

#### 3. Asymptotic Lower Bound
The paper proves that in the limit (as $N \to \infty$), Diversified Sampling is guaranteed to outperform Standard Sampling because it eliminates the fixed inductive bias of the model, eventually hitting the global minimum of the loss manifold.

---

### Analysis of Implementation Results


<img width="854" height="673" alt="image" src="https://github.com/user-attachments/assets/e15b4bfa-3898-4098-a32d-e221b8b16e9e" />

#### Graph 1: Inference Scaling Laws (MSE vs. Sample Budget)
The results demonstrate a clear inverse scaling relationship. As $N$ moves from $2^0$ to $2^6$ on the log-scale:
*   **Performance Gain:** The MSE floor drops significantly across all configurations.
*   **Temperature Synergy:** DSI performs best at a balanced temperature (0.7). At low temperatures (0.1), the model is too "rigid" to benefit from the shook input. At high temperatures (1.2), the results become too chaotic.
*   **Strategy Comparison:** The lines representing Algorithm 1 (Structural Shaking) consistently sit below standard noise, proving that "intelligent shaking" is more effective than random jitter.


<img width="839" height="676" alt="image" src="https://github.com/user-attachments/assets/04a33ac1-06df-4697-9dc2-35393bb7b136" />

#### Graph 2: Visual Fidelity (DSI Pro vs. Reality)
The visual comparison highlights the success of the Majority Voting (Median) aggregator.
*   **Error Correction:** While a standard forecast ($N=1$) misses sharp peaks in the Oil Temperature data (ETTh1), the DSI forecast (Green Line) successfully captures the trend shifts.
*   **Horizon Robustness:** Even though Chronos is optimized for a 64-step horizon, DSI allows it to maintain accuracy through a 96-step horizon by using the consensus of 64 diversified attempts to filter out individual model "hallucinations."

---

### Fail Cases and Constraints
*   **Suffix Padding:** Adding irrelevant data to the end of the history corrupts the model’s recent memory, leading to a performance collapse.
*   **Temperature Misalignment:** Without sufficient temperature, the model cannot explore the new diversity created by the perturbations.

### Summary
Diversified Scaling Inference proves that accuracy in Time Series AI is not just about the size of the model, but the amount of "thinking time" allocated during prediction. By using STL Decomposition to intelligently perturb the input and aggregating 64 candidate paths via the Median, we can break through the training limitations of foundation models and achieve state-of-the-art precision in zero-shot forecasting.
