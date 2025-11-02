# Selective State Space Models - SSSM

In SSMs we have a way to precompute everything and do the convolution but its not really good. It performs bad, since the $\Delta$ in each is THE SAME ! But it wont work good since A, B and C are being trained differently !

- So in SSSM, all the input tokens are being seen differently, so that the SSM can look at the token and drop certain tokens or ignore them or give them less attention. It would make it go further as well as be poductive !

- Here, we have different $B$, $C$ AND $\Delta$, in form of $B_0, B_1, B_2,...$, $C_0, C_1, C_2,...$ and $\Delta_0, \Delta_1, \Delta_2,...$
  > Note: now it can look at all tokens differently and decide what to give more attention to and what to not - which is something Attention did.

And now the dream of precalculating those is NOT possible.
We can not precompute since - $$y_k=C_0A^kB_0x_0+C_1A^{k-1}B_1x_1+C_2A^{k-2}B_2x_2+...+C_{k-1}AB_{k-1}x_{k-1}+C_kB_kx_k$$

But we can also ue a trick. We can do pareller Associative scan.

- for ex. [3 1 7 0 4 1 6 3] -> Prefix sum -> [0 3 4 11 11 15 16 22].
- We can do calculation of B and C on the go - but idk
- They do these in SRAM and not in HBM, after that they discretize the A and B and then perform `parallel assoiciative scan` yielding intermediate states of B and C with the size of (B, L, D, N) in SRAM.
- We multiply and sum with C, and produce the outputs of size (B, L D) and write it to HBM.
