# State Space Models (SSMs)

- The starting of the mamba.
- SSMs are better because of their linear calculation and time complexity in long sequences compared to quadratic for trasformers. But they were not as performant as trasformers and hence `mamba` was introduced.
- These SSMs work like a simple Linear RNNs.
- Also called `four models`, because of having 4 set of matrices to train it. Specially there is a $\Delta$, A, B and C.

  1. A determines, how much of hidden state to move forward.
  2. B determines, how much of input to take in.
  3. C determines, how much of output to be generated.
  4. $\Delta$ is what makes changes to A and B to make it $\bar{A}$ and $\bar{B}$

- I guess $\Delta$ is just backpropogation variable.

---

### Step - 1 - Discretization

- We use $\Delta$ to create $\bar{A}$ and $\bar{B}$. We train those matrices, A and B.

### Step - 2 - Linear RNN

- Here we enter the linear RNN phase.
- We use $h_{t-1}$ as hidden state from last state and the new input $x_t$ to get $h_t$ just like in traditional RNNs.
  $$h_t=\bar{A}h_{t-1}+\bar{B}x_t$$

- To get the final represenation($y_t$) of that token, we have another matrix C which takes care of it like-
  $$y_t=Ch_t$$
- these in terms of LLMs, can be logits !

---

Well, SSMs are linear RNNs. Not Traditional. Traditional RNNs have nothing as GPU parellalization. Linear do. But what is the difference.

> Linear RNNs have `NO` nonlinear activations like tanh and relu making it much faster per step. This makes the Linear RNNs to base PURELY on the matrix multiplication, which can be parallelized. And hence both of the above, gradients become simpler as well !

And to prove the above point (assuming $A=\bar{A}$ and $B=\bar{B}$),

- `First` token is $$h_0=Ah_{-1}+Bx_1=Bx_0$$ and hence the output for the `first` token comes out to be $$y_0=CBx_0$$

- Now for the `second`, $$h_1=Ah_0+Bx_1$$ $$h_1=ABx_0+Bx_1$$ and leading to $$y_1=Ch_1$$ $$y_1=ABCx_0+BCx_1$$.

- Similary for the `third`, $$h_2=Ah_1+Bx_2$$ $$y_2=CA^2Bx_0+CABx_1+CBx_2$$

- We see a pattern here !
  $$y_k=CA^kBx_0+CA^{k-1}Bx_1+CA^{k-2}Bx_2+...+CABx_{k-1}+CBx_k$$

- Now we already know the sequence length which in case here, is `k`. What we can do is, precompute 2 vectors- $$K=(CA^kB,CA^{k-1}B, CA^{k-2}B,...,CAB, CB)$$ and $$x=(x_0, x_1, x_2,..., x_{k-1}, x_k),$$ we can use these precomputed K matrix as well as the input matrix x, we can literally just combine them like - $$y=Kx.$$ AND Convolutions are flipping fast on GPUs.

- SSMs are faster and less computationally extensive ! but not so good at having performance !

> **Note**: A, B, C are learnable params and can be trained using backpropogation which are fized throughout.