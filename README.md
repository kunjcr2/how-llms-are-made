# **How LLMs are made (from scratch)?**

> A simple pipeline of how LLMs are made from scratch

## Data Preprocessing pipeline

We basically get the data from the file and convert those into encodings of tokens using **tiktoken** and then we convert those into the datasets and convert dataset to dataloader and then using embedding layer to get the embeddings of tokens as that is what we call feature extraction.

1. Get the data, and store it in a variable
2. Use tiktoken and get the vocab of gpt2 which is about 50257 in length.
   - We use `tiktoken.get_encoding("gpt2")`
3. Now create a dataset class and have input_ids and output_ids and using max_length and stride varibales, we make input_ids and target_ids of certain length, usually it is called the context_length. And dont forget to keep those in tensors.
4. create a dataloader function to convert the dataset into dataloaders of certain batch_size.
5. Now we basically create embedding layer to get the token embeddings using (vocabulary_dimension, output_dimension_of_vector) as arguments.
   - Embedding layer is basically a lookup table of EVERYTHING on vocabulary with random values initialized.
   - And we get the embeddings from this based on the encoding we pass in the layer.
   - These weights are trained during training.
   - Total number of weights = Vocab size \* embedding vector dimension
6. And we pass in the input_ids from dataset class to embedding layer, and that will create the token_embeddings.
7. Now we create another embedding layer with (context_vector_length, output_dimension_of_vector) for positional_embeddings.
8. Now we basically add token_embeddings and positional_embeddings to create input_embeddings which are input to the model.

## Attention Mechanism

We need some sort of attention mechanism as only having the meaning of the word and the position wouldnt work and that creates issues with long sequence dependencies. RNNs worked but not for long term. We use attention as attention is basically how important a word is compared to the other words in the sequence or how much attention should be given to other words in the sequence.

1. We use simplified Attention mechanism at the BASE level. Basically we get the attention scores (w) using dot products of each token with every other token and then apply softmax to it and multiply w with corresponding input embedding to get context embeddings.
2. In the self attention, we have weights of query, key, value matrices which are multiplied to input tensor to get values, keys, queries tensors. We get attention scores by doing `queries@keys.T` which are then normalized by softmax after dividing by `sqrt(d_of_keys)`. After getting attention_weights, we do `attn_weights@values` to get the context vectors.
3. In the next step, we dont let the model look at the future tokens and for that we hide the future tokens and let it see past and present tokens ONLY. We do this by creating a buffer of upper trainglur matrix, and whereever there is a one, we change it to -infinity, after which we do softmax of attention scores which basically makes the future token's attention score = 0. We apply dropout for better generilazation. This is Casual Attention.
4. In Multi-headed attention, we use multiple Casual Attentions instances using `nn.ModuleList`, and passing the batches through it to get multiple context vectors and concating it column wise. We basically create Multiple Head or multiple single casual attention instances and run everything in sequential.
5. In the final showdown, we use bunch of stuff like `num_head` and `head_dim` to convert ALL the weight matrices in a one big matrice and after undergoing same set of rules from `3.`, we come across bunch of context vectors from various heads, which are concated at the end to get one big context vector for a token. This is better as we are making the matrix multiplication MUCH efficient as we're doing just ONE for each Q,K,V compared to few in `4.`. For more info, refer to notes or `./attention/LLM_attention.ipynb`.

## LLM Architecture

1. We talk about the actual workflow of how the data passes in LLM. We go from input layer to transformer block which contains Normalization layer, Multi head attention mechanims and then feed forward neural netowrks and much more. After which we move to the output layers and get logits. We will be training GPT2-124M model from scratch.
2. We talked about how the Layer normalization work and what is it, and why is it used to prevent exploding and vanishing gradient. We basically try to make the mean = 0 and variance = 1, by substituting `x` to
   `scale*((x_i-mean)/sqrt(variance+eps))+shift`. Scale and shift are basically for having smoother training. Also, gradient basically how should the weight be changed to minimize the loss.
3. WE talk about feed forward network as well as the GeLU activation function which is a better version of ReLU function and clears all the issues. We basically return
   `0.5*x*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))* (x + 0.044715*x**3)))`, and it works actually between 2 linear layers with neurons benig four times the embedding dim. They are stackable.
4. Gradient vanishing is a bigg issue, and just to solve it, we use shortcut connection where we basically add output with input of the layer and that preserves the gradient from vanishing !
