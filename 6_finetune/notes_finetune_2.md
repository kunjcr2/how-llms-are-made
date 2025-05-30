We learn about **Instructional Finetuning**.

## Lec1 - Creating datasets

- Pretrained models are good at text completions, but not at instructions.
- We use the dataset which looks like below.

```json
{
  "instruction": "here_comes-the_instruction",
  "input": "here_comes-the_input",
  "output": "here_comes_the_output"
}
```

And millions nd billions of the datasets !

And we convert those to -

- We use `AlphaCa prompt` stype by stanford as it MUCH more common. Search up to google to check what that is and how it looks. Another one is `Phi3 prompt`.

---

## Lec2 - LoRA - Shaw Talebi

> 1. **LoRA** - Fine tuning by adding new trainable params

- Lets say input is `x` and hidden layer is `h(x)`. So basically, `h(x) = W0*x` where we have the hidden layer by multiplying the input with the Weight matrix. This is what we have for "**Not LoRA**".
- In LoRA, we have `h(x) = W0*x + del(W)*x`. We have these additional weight matrix. Lets assume `del(W) = BA`. `h(x) = W0*x + BA*x`, so we have something like B an A being a vector and and we multipley them and add it to weight matrix and multiply with x to get hidden layer.
  - We have W0, frozen here and B and A are trainable with Far less params compared to the ones before
