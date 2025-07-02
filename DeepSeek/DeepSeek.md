> # How was DeepSeek built ?

---

## What is DeepSeek ?

1. Lec-1 was basically what is Deepseek and what we will be looking at in the course.
2. Lec-2 is about why is it special with MoE, MTP, MLA, Quantization, RoPE and all that. We also talked about what Reinforcement learning. Instead of human based labeled data, they used RL which basically taught the model to do complex reasoning. RL is basically rule based reward system.

---

## Starting now with DeepSeek

3.  Lec 3, 4, 5, 6, 7 and 8 are just Multi Head Attention Mechanism.
4.  `Key-Value cache` or `KV Caching` (Popular term). Lets say we have a next token generation task.
    a. AB -> Inference -> C
    b. ABC -> Inference -> D
    c. ABCD -> Inference -> E

    - So here what happens is that, we have to calculate K, Q, V, attn scores and attn weights, in transformer block for AB, thrice; ABC, twice. Which is basically waste of computation.
    - Also, as we know, to predict last token, we need the context vector of the last token ONLY. Lets assume that last token embeddings to be X.

      ```
      X*Wq=Q, X*Wk=K, X*Wv=V
      ```

      but now the the thing is We are getting K of last token. What about others. Thats where we get the remaining K and V matrices from cache which we already have.

      So we take that cached K and V and use it to get the attn scores, attn weights and ahead with not having any sort of redundant calculations.

      We basically need Context vector of last token, which can be recieved with attention weight of last token and Value matrix. We get Value vector of last token from XxWv and all we do is add other Value matrix parts from cache. Same while getting attn weights, we multiply Query vector with transpose of Key matrix. Where we get Key's final vector from XxWk and the rest from cache.

5.  We use MultiQuery attention, where we make all the attention heads of keys an values, SAME.
    - In Normal attention, we have multiple attention heads for a single Key or Value matrix, and all of them are different but we make all of them same so that we ONLY have to save that single head which can be concated and used as an entire Key Value matrix.
    - Plus point of this is, we reduce 400 GB of KV Caching, to 3 GB by not saving all the attention heads.
    - The only downside is we loose performance as multiple heads means multiple perspective but now only 1.
