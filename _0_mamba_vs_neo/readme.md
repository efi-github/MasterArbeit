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

from https://huggingface.co/intfloat/e5-mistral-7b-instruct/blob/main/lora/adapter_config.json


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
(from mamba paper)


The gpt-neo model was trained at a constant learning rate of:
{
    "n_head": 32,
    "n_vocab": 50257,
    "embed_dropout": 0,
    "lr": 0.0002,
    "lr_decay": "cosine",
    "warmup_steps": 3000,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-8,
    "opt_name": "adam",
    "weight_decay": 0.1,
    "train_batch_size": 512,
    "attn_dropout": 0,
    "train_steps": 286150,
    "eval_steps": 10,
    "predict_steps": 1,
    "res_dropout": 0,
    "eval_batch_size": 512,
    "predict_batch_size": 1,
    "iterations": 500,
    "n_embd": 2048,
    "datasets": [["pile", 25, "documents_random", 1.0]],
    "model_path": "gs://neo-models/GPT3_XL_Pile",
    "n_ctx": 2048,
    "n_layer": 24,
    "scale_by_depth": true,
    "scale_by_in": false,
    "attention_types" :  [[["global"],24]],
    "mesh_shape": "x:128,y:2",
    "layout": "batch:x,memory_length:y,embd:y",
    "activation_function": "gelu",
    "recompute_grad": true,
    "gradient_clipping": 1.0,
    "tokens_per_mb_per_replica": 2048,
    "precision": "bfloat16"
}
(from https://github.com/EleutherAI/gpt-neo#training-guide and https://github.com/EleutherAI/gpt-neo/blob/master/configs/gpt3_XL_256_Pile.json)