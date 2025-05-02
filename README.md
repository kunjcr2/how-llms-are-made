# **How LLMs are made ?**

> A simple pipeline of how LLMs are made from scratch

## Data Preprocessing pipeline

We basically get the data from the file and convert those into encodings of tokens using **tiktoken** and then we convert those into the datasets and convert dataset to dataloader and then using embedding layer to get the embeddings of tokens as that is what we call feature extraction.

1. Get the data, and store it in a variable
2. Use tiktoken and get the vocab of gpt2 which is about 50257 in length.
   - We use .get_encoding("gpt2")
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
