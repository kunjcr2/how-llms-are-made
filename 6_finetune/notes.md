Whenever we downlaod those 7 files from lets say, kaggle. We have a function `load_gpt2_params_from_tf_ckpt()` in the script `gpt_download3.py`, which returns a disctionary of parameters. And it has 5 keys:

1. wte
   - Token Embeddings
2. wpe
   - Positional Embeddings
3. Blocks
   - Attention blocks with Q, K, V
   - Feed forward neural network weights
   - Output projection which gives logits
   - Layer normalization scale and shift for both of the kinds
     > Similar for all transformer blocks
4. Final normalization scale
5. Final normalization shift

---

There is basically a class and weights to be downloaded from internet or from kaggle which, by using some sort of processing, and then we can use those.

---

Go to `https://youtu.be/yXrGeDNuymY?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu`.

- This is a bit complex, you can go the video and get the idea of what are we doing. This is much complex than what I expected it to be. I am so sorry for being lazy !
