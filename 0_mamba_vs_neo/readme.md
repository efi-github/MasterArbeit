## Mamba vs Neo

In this small project we compare the ability of both mamba and gpt-neo 
in classification using lora adapters.

An inspiration was the https://huggingface.co/intfloat/e5-mistral-7b-instruct model,
which has been able to achieve high classification/embedding results.

For these tests we use the same lora weights as in the above model:
- r : 16
- lora_alpha : 32
- lora_dropout : 0.1

It is irrelevant for mamba, but the mistral model targeted the following modules:
- "q_proj",
- "k_proj",
- "v_proj",
- "o_proj",
- "down_proj",
- "up_proj",
- "gate_proj"

The original mamba model was trained like this:
AdamW optimizer with
• gradient clip value 1.0
• weight decay 0.1
• no dropout
• linear learning rate warmup with cosine decay
By default, the peak learning rate is the GPT3 specification.
We give several models an “improved recipe”, inspired by changes adopted by popular large language models such
as PaLM (Chowdhery et al. 2023) and LLaMa (Touvron et al. 2023). These include:
• linear learning rate warmup with cosine decay to 1−5, with a peak value of 5× the GPT3 value
• no linear bias terms
• RMSNorm instead of LayerNorm
• AdamW hyperparameter  = (.9, .95) (the GPT3 value) instead of the PyTorch default of  = (.9, .999)

The gpt-neo model was trained at a constant learning rate of:
