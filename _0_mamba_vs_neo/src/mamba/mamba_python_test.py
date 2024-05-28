#%%
from _0_mamba_vs_neo.models.MambaForSequenceClassification import MambaForSequenceClassification
import _0_mamba_vs_neo.datasets.ecthr.utils_ecthr as utils_ecthr
#%%
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss
import os
#%%
os.environ["WANDB_PROJECT"] = "mamba_vs_neo"
#%%
"""
CONFIGS:
"""
#%%
"""
    general:
        - RUN_NAME: str
            name of the run
        - OUTPUT_DIR: str
            directory to save the model and logs
        - SEED: int
            random seed to use
        - REPORT_TO: str
"""
RUN_NAME = "mamba_testing"
OUTPUT_DIR = f"_0_mamba_vs_neo/models/mamba/{RUN_NAME}"
SEED = 42
REPORT_TO = "wandb"
#%%
"""
    dataset:
        - ALLEGATIONS: bool
            True: use allegation data for the cases, so what laws did the cases allegedly violate
            False: use court decisions, so what laws did the court decide the cases violated
        - SILVER: bool
            True: only use facts which were deemed relevant by the court
            False: use all facts
        - MULTI_LABEL: bool
            True: use multi-label classification (which law was (allegedly) violated)
            False: use binary classification (was there a law (allegedly) violated)
        - FREQUENCY_THRESHOLD: int
            minimum number of cases a law must be (allegedly) violated in to be considered
        - NUM_LABELS: int
            number of labels in the dataset (ecthr: 41)
        - MAX_LENGTH: int
            maximum number of tokens in a sequence     
"""
ALLEGATIONS = True
SILVER = False
MULTI_LABEL = True
FREQUENCY_THRESHOLD = 0
NUM_LABELS = 41

MAX_LENGTH = 512
#%%
"""
    training:
        - EPOCHS: int
            number of times to iterate over the dataset
        - LEARNING_RATE: float
            rate at which the model learns
        - BATCH_SIZE: int
            number of sequences in a batch
        - GRADIENT_ACCUMULATION_STEPS: int
            number of batches to accumulate gradients over
        - USE_LENGTH_GROUPING: bool
            True: group sequences of similar length together to minimize padding
            False: do not group sequences by length
        - WARMUP_RATIO: float
            ratio of training steps to warmup steps
        - MAX_GRAD_NORM: float
            maximum gradient norm to clip to
        - WEIGHT_DECAY: float
            weight decay to apply to the model
"""
EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
print("true batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 0.3
WEIGHT_DECAY = 0.001

USE_LENGTH_GROUPING = True
#%%
"""
    evaluation:
        - EVAL_STEPS: int
            number of steps between evaluations
        - BATCH_SIZE_EVAL: int
            number of sequences in a batch for evaluation
        - LOGGING_STEPS: int
            number of steps between logging
        - EVAL_ACCUMULATION_STEPS: int
            number eval batches to calculate before copying to the cpu, if the eval requires a lot of memory this is helpful
"""
EVAL_STEPS = 200
BATCH_SIZE_EVAL = BATCH_SIZE
LOGGING_STEPS = 100
EVAL_ACCUMULATION_STEPS = 20
#%%
"""
    model:
        - MODEL_NAME: str
            name of the model to use
        - LORA_TASK_TYPE:
        - LORA_R: int
           r is the rank of the approximation
        - LORA_TARGET_MODULES: list
            list of modules to target with LoRA
"""
MODEL_NAME = "state-spaces/mamba-1.4b-hf"
LORA_TASK_TYPE = TaskType.SEQ_CLS
LORA_R = 8
LORA_TARGET_MODULES = ["x_proj", "embeddings", "in_proj", "out_proj"]
#%%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)
    
    precision_macro, recall_macto, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, predictions, average='micro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)

    return {
        'strict_accuracy': accuracy,
        'hamming_accuracy': 1 - hamming_loss(labels, predictions),
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'precision_micro': precision_micro,
        'recall_macro': recall_macto,
        'recall_micro': recall_micro
    }
#%%
class SimpleBCELossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = torch.nn.BCEWithLogitsLoss()
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss
#%%
model = MambaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
#%%
tokenizer.pad_token_id = tokenizer.eos_token_id
model.pad_token_id = tokenizer.eos_token_id
#%%
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding = True)
#%%
ecthr_dataset = utils_ecthr.load_ecthr_dataset(allegations=ALLEGATIONS, silver=SILVER, is_multi_label=MULTI_LABEL, frequency_threshold=FREQUENCY_THRESHOLD)
ecthr_dataset = utils_ecthr.tokenize_dataset(ecthr_dataset, tokenizer, max_length=MAX_LENGTH)
ecthr_dataset = ecthr_dataset.remove_columns("facts")
#%%
train = ecthr_dataset["train"]
val = ecthr_dataset["validation"]
test = ecthr_dataset["test"]
#%%
lora_config =  LoraConfig(
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        task_type=LORA_TASK_TYPE,
        bias="none"
)
#%%
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
#%%
model.to("cuda")
#%%
#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(name)
#%%
training_args = TrainingArguments(
    output_dir= OUTPUT_DIR,
    run_name= RUN_NAME,
    learning_rate= LEARNING_RATE,
    lr_scheduler_type= "constant",
    warmup_ratio= WARMUP_RATIO,
    max_grad_norm= MAX_GRAD_NORM,
    per_device_train_batch_size= BATCH_SIZE,
    per_device_eval_batch_size= BATCH_SIZE_EVAL,
    gradient_accumulation_steps= GRADIENT_ACCUMULATION_STEPS,#
    group_by_length= USE_LENGTH_GROUPING,
    num_train_epochs= EPOCHS,
    weight_decay= WEIGHT_DECAY,
    eval_strategy="steps",
    eval_steps= EVAL_STEPS,
    eval_accumulation_steps = EVAL_ACCUMULATION_STEPS,
    save_strategy="steps",
    save_steps= EVAL_STEPS,
    load_best_model_at_end=True,
    report_to= REPORT_TO,
    fp16=False,
    gradient_checkpointing=True,
    logging_dir="logs",
    logging_steps= LOGGING_STEPS,
    label_names=["labels"],
)
#%%
#%%
trainer = SimpleBCELossTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=train,
    eval_dataset=val,
    compute_metrics=compute_metrics
)
#%%
trainer.train()
#%%
trainer.evaluate(test)
#%%
predictions = trainer.predict(test)
#%%
def calulate_metrics_index(predictions, index):
    logits = predictions.predictions
    labels = predictions.label_ids
    
    logits = logits[:, index]
    labels = labels[:, index]
    
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    
    count_correct = np.sum(labels)
    count_predicted = np.sum(predictions)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'count_cases': count_correct,
        'count_predicted': count_predicted
    }

#%%
ids = utils_ecthr.ARTICLES_ID
ids = {v: k for k, v in ids.items()}
desc = utils_ecthr.ARTICLES_DESC

#%%
# starting at 1 because 0 is not occupied due to an indexing error
for i in range(1, 41):
    print("-"*50)
    print(f"Label {i}")
    print(ids[i])
    print(desc[ids[i]])
    print(calulate_metrics_index(predictions, i))
#%%

