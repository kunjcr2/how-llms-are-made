# Mixture of Experts (MoE) Architectures

## Overview
Mixture of Experts (MoE) architectures are a key technique powering many advanced AI models. They enable dramatic scaling of model parameters without slowing down training or inference speed by routing each token to only a sparse set of expert networks.

## The Role of Feedforward Networks (FFN)
In a standard Transformer block, attention mechanisms capture contextual dependencies, while FFNs run independently for each token to retrieve factual information embedded within its weights.
- **Normalization (RMSNorm):** Before the FFN, a normalization layer scales the token vector to unit norm with a learnable scaling factor per dimension to prevent instability.
- **Up-projection ($W_{up}$):** A linear layer maps the input vector to a higher-dimensional space (typically 4x larger). Each learned direction represents a semantic concept or question posed to the token.
- **Activation (e.g., ReLU):** Removes irrelevant semantic matches (negative values).
- **Down-projection ($W_{down}$):** Maps the activated vector back to the original dimension. The column vectors encode factual information corresponding to the concepts selected by the activation. This output is added to the original embedding via a residual connection.

Increasing the hidden dimension of the FFN captures more knowledge but increases computation and memory. MoE solves this by recognizing that for any given token, many hidden units are irrelevant. 

## Sparse Mixture of Experts
MoE partitions a large FFN into several smaller, specialized **expert networks**. For each token, only a sparse subset of experts is activated. This creates a highly expressive network without incuring additional computational costs at test time.

### Router Design
A router predicts which experts to activate for a given token.
1. The router computes a score for each expert using learned weight vectors $g_i$.
2. It takes the dot product between the token embedding $x$ and $g_i$ to produce logits $h(x)$.
3. A softmax function normalizes these logits into probabilities $p(x)$.
4. The router selects the top-$k$ experts (e.g., top 2) with the highest probabilities.
5. The final output is a weighted sum of the chosen experts' outputs, using the routing probabilities as weights.

### Expert Variations
- **Fine-grained Experts:** Instead of a few large experts, recent models (e.g., DeepSeek V3, Qwen 3) use many smaller experts (e.g., 256), activating a subset (e.g., 8). This allows finer granularity and better performance.
- **Shared Experts:** A dedicated expert that is always active for every token. It learns broad, common knowledge, allowing routed specialists to focus on nuanced patterns. (Effectiveness is debated; some models show improvements, others do not).

## Token-to-Expert Assignment & Imbalance
Routing is dynamic, which can lead to load imbalances across devices and experts.

### Token Overflow and Dropless MoE
- **Expert Capacity:** $ \text{Expert Capacity} = \frac{\text{Total Tokens in Batch}}{\text{Number of Experts}} \times \text{Capacity Factor} $
- **Token Overflow:** When an expert exceeds its token capacity, overflowing tokens are dropped (computation skipped, passed unchanged via residual connection).
- **Tradeoff:** Increasing the capacity factor reduces overflow but wastes computation/memory on padding empty slots.
- **Dropless MoE:** Avoids this tradeoff by representing expert computation as block-diagonal matrix multiplications. With variable block sizes (depending on the number of tokens routed to each expert), block sparse matrix multiplication kernels process all tokens without dropping or padding.

## Load Balancing
Since experts are initialized randomly, unbalanced routing early in training creates a feedback loop: frequently used experts learn quickly and are preferred more, while unused experts become "dead".

### Load Balancing with Auxiliary Loss
To ensure tokens are distributed evenly:
1. **Noisy Top-K Gating:** Adding tunable Gaussian noise to logits encourages diverse routing.
2. **Load Balancing Loss:** Soft constraints are added to the loss function.
   - **Importance:** Sum of router probabilities for an expert over a batch. A balanced system has a low coefficient of variation (CV) for importance.
     - *Coefficient of Variation (CV):* Quantifies variability by calculating the ratio of the standard deviation to the mean. A lower CV indicates expert importance is distributed more evenly. When importance is perfectly uniform, the CV becomes zero.
   - **Load:** The actual number of tokens assigned to an expert. (Note: CV for importance can be zero even if actual token loads are unbalanced, so load must be measured as well). Since this raw count is non-differentiable, a smooth estimator (probability of selection) is used.
   - The simplified auxiliary load balancing loss encourages both the average router probability per expert ($\pi_i$) and the fraction of tokens per expert ($f_i$) to approach $1/N$. It is weighted by a hyperparameter $\alpha$.

### Auxiliary-Loss-Free Load Balancing (DeepSeek V3 approach)
Auxiliary losses force a tradeoff between model performance and load balance. DeepSeek V3 avoids this:
- Instead of softmax, it uses a sigmoid function on router logits.
- A dynamic, small bias term is added to each expert's logits during top-$k$ routing calculation.
- If an expert is overloaded, its bias is decreased. If under-loaded, its bias is increased.
- The top-$k$ selection uses these bias-adjusted scores, but the final output combination uses the original probabilities, maintaining both load balance and model performance.

## Stabilizing Training with Router Z-Loss
The router uses a softmax function, which is **shift-invariant** (adding a constant to all logits doesn't change outputs). However:
- This allows the underlying logits to grow indefinitely during training.
- In low precision (Float16), this can quickly cause numerical overflow and training collapse.
- **Safe Softmax** (subtracting the max logit) prevents immediate overflow but doesn't stop logits from growing.
- **Router Z-Loss:** Adds a penalty for large log-normalization terms. It penalizes both positive and negative shifts by trying to keep the logits centered and small. 
- Using Router Z-Loss prevents gradient explosions and ensures stable training natively, overcoming the loss spikes found in unregularized MoE training.

## Conclusion
Modern State-of-the-Art MoE training involves three main components:
1. Standard cross-entropy loss for next-token prediction
2. Load Balancing Loss (or auxiliary-loss-free bias adjustment) to prevent dead experts
3. Router Z-Loss to keep training numerically stable
