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

      - But issue with that was, it takes too much of space.

      ```
        size of KV cache = l*b*n*h*s*2*2
        where,
        l is number of transformer blocks
        b is batch size
        n is numebr of heads
        h is head dimension
        s is context length
        2 is one for Key and one for Value
        2 is float16 bit number = 2 byte
      ```

      and this leads to Deepseek being 400GBs at storing KV cache in memory and hence, this method is sh\*t.

5.  We use `MultiQuery attention`, where we make all the attention heads of keys an values, SAME.

    - In `Normal attention`, we have multiple attention heads for a single Key or Value matrix, and all of them are different but in `MultiQuery attention` make all of them same so that we ONLY have to save that single head which can be concated and used as an entire Key Value matrix.
    - Plus point of this is, we reduce 400 GB of KV Caching, to 3 GB by not saving all the attention heads.
    - Minus point is we loose performance as multiple heads means multiple perspective but now only 1.
    - Here, KV Cache formula is reduced to `l*b*h*s*2*2` as we dont care about number of heads anymore.

6.  In `Grouped Query Attention`, it is somewhere between `MHA` and `MQA`. `MHA` have all the K and V heads as distinct matrices, leading to KV Caching memory issue. While in `MQA`, they ALL are same, leading to loss in accuracy.

    - what we do here in `GQA`, is we make small groups of heads, for example out of 64 heads, we make 8 groups of 8 heads. Now in 8 groups, each group have distinct K-V matrices but within each group, all the heads have common K-V matrices.

    ```
    [1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4] - 4 groups with 4 heads each
    ```

    Above each 1, 2, 3, 4 are K-V amtrices which are same within the group but distinct for each group.

    - Our KV Cache formula is now `l*b*g*h*s*2*2` where g is just number of groups.
    - This leads to memory requirement of GPT-3 which is,

    ```
    MHA - 4.5 GBs
    MQA - 48 MBs
    GQA - 384 MBs (which is not much but does its job)
    ```

    - Better performance than `MQA`, but as not good as `MHA`.

> ## And that is where deepseek comes in which solved the problem using `Multihead Latent Attention`, where they got K-V memory requirement to low and Performance at High.

### 7. **`Multihead Latent Attention`**

- Alright. So what we did here is,
  1. We take the input embeddings `X`. We do `X*Wq` to get `Q` vector.
  2. We do `X*Wdkv`, to get a Latent matrix `Ckv` which is to be cached, where `Wdkv` is a down projection of input into cached latent matrix.
  3. We do `Ckv*Wuk` to get `K` vector, where `Wuk` is up projection of Key.
  4. We do `Ckv*Wuv` to get `V` vector, where `Wuv` is up projection of Value matrix.
  5. Then getting attn scores, attn weights and finally context vector.

```
context_vector
    = attn_weights * V
    = Q*K.T*V
    = (X*Wq)*(Ckv*Wuk)*(Ckv*Wuv)
    = (X*Wq)*(X*Wdkv*Wuk).T*(X*Wdkv*Wuv)
    = X * Wq * Wuk.T * Wdkv.T * X.T * X * Wdkv * Wuv
and that's exactly where we get 'Absorbed Query'.
```

- Basically instead of caching both, Keys and values, we cache a common matrix from where we can have both Keys and Values vector.
- PLus point is, the formula now becomes,

```
Size of cache = l*b*nl*s*2
    where,
    l is transformer blocks
    b is batch numbers
    nl is dimension of cached matrix
    s is context length
    2 is 2 bytes = 16 bits
```

- Basically, deepseek becomes 6GBs rather than those big numbers. Now, the memory issue is solved. Coming to perfomance. All the `Wuk` and `Wuv` are there already, FIXED, but cached `Ckv` comes in everytime frmo cache and is being updated everytime leading to distinct heads and hence better performance.
