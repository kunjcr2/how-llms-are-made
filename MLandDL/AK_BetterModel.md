### He mentioned some of the points to make the training faster. And by faster, he **MEANS** it.

- **The biggest exchange of data and loss of time happens between travel of data and not calculations.**

1. Parameter sharing between Embed and UnEmbed layers.
2. Model weights initialized by having normal STD and Variance. We also do something with NUmber of residual connections \* some penalty. (I dont know why but we do it. For more info, check [video](https://youtu.be/l8pRSuU81PU))
3. Using `torch.cuda.synchronize()` which will wait for GPU to get done with the task before moving ahead. More like async programming. It reduces GPU overload.
4. Using GPUs. Using `Mixed Precision Training`. It means we use `float16` or `bfloat16` instead of default `float32` reducing a large amount of memory and making it faster. Even `TFfloat32` is significantly fast. We use-

   ```python
   from torch.cuda.amp import autocast

   with torch.autocast(device="cuda", dtype=torch.bfloat16):
       logits = model(input)
       loss = loss_fn(logits, targets)

   # We just keep model and loss function in here, not .step or .backward, as it would create issues.
   # We do those in classic FP32 or default.
   ```

5. Now comes the MOST POWERFUL TOOL I HAVE HEARD SO FAR - `torch.compile()`. We do-

   ```python
   model = torch.compile(model) # This is SO OVERPOWERED
   ```

   It fuses the kernals and reduces the Python overload. Before the interpreter converted each line of code and ran it which was pretty messed up, but now with the already compiled model, runs pretty fast as the GPU now knows what code is and it reduces HBM to SM travel. It reduces from about 7-8 travels to just 1-2.

6.
