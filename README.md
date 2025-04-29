# **How LLMs are made ?**
> A simple pipeline of how LLMs are made from scratch

## Data Preprocessing pipeline
1. Get the data, and store it in a variable
2. Use tiktoken and get the vocab of gpt2 which is about 50257 in length.
3. Now create a dataset class and have input_ids and output_ids and using max_length and stride varibales, we make input_ids and target_ids of certain length, usually it is called the context_length. And dont forget to keep those in tensors.
4. create a dataloader function to convert the dataset into dataloaders of certain batch_size.
5. Now we basically create embedding layer to get the token embeddings using (vocabulary_dimension, output_dimension_of_vector) as arguments.
   - These weights are trained during training.
   - And they are initialized randomly
   - Total number of weights = Vocab size * embedding vector dimension
6. And we pass in the input_ids from dataset class to embedding layer, and that will create the token_embeddings.
7. Now we create another embedding layer with (context_vector_length, output_dimension_of_vector) for positional_embeddings.
8. Now we basically add token_embeddings and positional_embeddings to create input_embeddings which are input to the model.
