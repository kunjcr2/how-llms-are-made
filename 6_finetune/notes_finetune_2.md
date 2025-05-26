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
