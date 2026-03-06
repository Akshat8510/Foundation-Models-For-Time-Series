# 📑 Paper 01: The o1 Moment for Time Series
**Title:** Diversified Scaling Inference in TSFMs (Jan 2026)

### The Main Idea
In the world of LLMs, we have **OpenAI’s o1**, which "thinks" before it speaks to get a better answer. This paper does the exact same thing for Time Series. 

Usually, we ask a model like **Chronos** or **TimesFM** for one forecast, and we just pray it’s right. This paper says: *"Don't ask once. Ask 64 times, but change the input slightly every time to see if the model converges on a better truth."*

---

### The Core Topics

#### 1. Pre-training vs. Inference Scaling
*   **Pre-training (The Old Way):** Spending millions of dollars to train a "Foundation" model on 100 billion data points.
*   **Inference Scaling (The New Way):** Taking a model that already exists and spending more "GPU time" during the prediction phase to squeeze out more accuracy.
*   **The Result:** A tiny model (Chronos-Tiny) can actually beat a huge model (Chronos-Large) if you give the tiny model more "thinking time" (Inference Scaling).

#### 2. The "Loaded Die" Analogy (Standard vs. Diversified)
*   **Standard Sampling:** Imagine a loaded die. If you roll it 100 times, you just get the same biased result over and over. This is what happens if you just sample a model with a fixed setting.
*   **Diversified Sampling (The Innovation):** Instead of rolling one die, you roll 100 different dice. You "shake up" the input data so the model has to look at the problem from different angles.

#### 3. Perturbation Strategies (The "How")
How do you "shake up" the input? The paper uses these methods (Keep these in mind for your code!):
*   **Gaussian Noise:** Adding "jitter" to the past data. (Like looking at a picture through a slightly rainy window).
*   **Masking:** Hiding some parts of the history. (Asking the model to guess the future while "blinking" at the past).
*   **Padding:** Adding zeros or extra data at the start.
*   **Task-Specific (The Pro Way):** Using **STL Decomposition** (Trend + Seasonality + Residual) to perturb specific parts of the wave.

#### 4. The Aggregators (The "Judge")
Once you have 64 different forecasts, how do you pick the winner?
*   **Majority Voting (MV):** You take the **Median** of all 64 tries. It’s the "Wisdom of the Crowd." This is usually the best for real-world stability.
*   **Exact Match (EM):** You pick the one that would have been the most accurate (The "Oracle" view). This shows the model's maximum potential.

---

### The Models Used
You need to remember these four for your codebase:
1.  **Chronos (Amazon):** The "Language" model for time series (uses T5).
2.  **TimesFM (Google):** The "Patch" model (decoder-only).
3.  **Moirai (Salesforce):** The "Universal" model (can handle any data frequency).
4.  **Time-MoE:** The "Expert" model (uses different sub-networks for different patterns).

---

### The New Metric: RobustMSE
Standard MSE is like checking a student's grade on one single test. **RobustMSE** is like taking the average of 5 tests where the questions were slightly different. It measures how **reliable** the model is when it’s allowed to "think" multiple times.

---

### The Examples
*   **The Fail Case:** The paper warns that **Suffix Padding** (adding junk to the end of the history) is bad—it makes models "hallucinate" and the error goes up.
*   **The Sweet Spot:** Diversified sampling works best when the "Cosine Similarity" is between **0.95 and 1.00**. If you change the input too much (too much noise), the model gets confused and gives you garbage.

---

### Summary
*"In this research, I analyzed how to trade compute for accuracy. I learned that by using Gaussian noise and Median aggregation, we can make open-source models like Chronos significantly more robust without needing to retrain them."*
