> # How was DeepSeek built ?

---

## What is DeepSeek ?

1. Lec-1 was basically what is Deepseek and what we will be looking at in the course.
2. Lec-2 is about why is it special with MoE, MTP, MLA, Quantization, RoPE and all that. We also talked about what Reinforcement learning. Instead of human based labeled data, they used RL which basically taught the model to do complex reasoning. RL is basically rule based reward system.

---

## Starting now with DeepSeek

3.  Lec 3, 4, 5, 6, 7 and 8 are just Multi Head Attention Mechanism.
4.  `Key-Value cache` or `KV Caching` (Popular term). Lets say we have a next token generation task.

    ```
    a. AB -> Inference -> C

    b. ABC -> Inference -> D

    c. ABCD -> Inference -> E
    ```

    - So here what happens is that, we have to calculate K, Q, V, attn scores and attn weights, in transformer block for AB, thrice; ABC, twice. Which is basically waste of computation.
    - Also, as we know, to predict last token, we need the context vector of the last token ONLY. Lets assume that last token embeddings to be X.

      ```
      X*Wq=Q, X*Wk=K, X*Wv=V
      ```

      - But now the the thing is We are getting K of last token. What about others. Thats where we get the remaining K and V matrices from cache which we already have.

      - So we take that cached K and V and use it to get the attn scores, attn weights and ahead with not having any sort of redundant calculations.

      > - We basically need Context vector of last token, which can be recieved with attention weight of last token and Value matrix. We get Value vector of last token from XxWv and all we do is add other Value matrix parts from cache. Same while getting attn weights, we multiply Query vector with transpose of Key matrix. Where we get Key's final vector from XxWk and the rest from cache.

5.  We use `MultiQuery attention`, where we make all the attention heads of keys an values, SAME.

    - In `Normal attention`, we have multiple attention heads for a single Key or Value matrix, and all of them are different but in `MultiQuery attention` make all of them same so that we ONLY have to save that single head which can be concated and used as an entire Key Value matrix.
    - Plus point of this is, we reduce 400 GB of KV Caching, to 3 GB by not saving all the attention heads.
    - Minus point is we loose performance as multiple heads means multiple perspective but now only 1.

6.  In `Grouped Query Attention`, it is somewhere between `MHA` and `MQA`. `MHA` have all the K and V heads as distinct matrices, leading to KV Caching memory issue. While in `MQA`, they ALL are same, leading to loss in accuracy.

    - what we do here in `GQA`, is we make small groups of heads, for example out of 64 heads, we make 8 groups of 8 heads. Now in 8 groups, each group have distinct K-V matrices but within each group, all the heads have common K-V matrices.

    ```
    [1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4] - 4 groups with 4 heads each
    ```

    Above each 1, 2, 3, 4 are K-V amtrices which are same within the group but distinct for each group.

    - This leads to memory requirement of GPT-3 which is,

    ```
    MHA - 4.5 GBs
    MQA - 48 MBs
    GQA - 384 MBs (which is not much but does its job)
    ```

    - Better performance than `MQA`, but as not good as `MHA`.

> ## And that is where deepseek comes in which solved the problem using `Multihead Latent Attention`, where they got K-V memory requirement to low and Performance at High.

7. **`Multihead Latent Attention`**
