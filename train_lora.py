import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Config
max_seq_length = 512
model_name = "mistralai/Mistral-7B-v0.1"
output_dir = "kz_lora_unsloth"

gc.collect()
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset("json", data_files="grammar_dataset.jsonl", split="train")


def tokenize_function(example):
    text = f"[INST] {example['instruction']} [/INST] {example['response']} </s>"

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # SFTTrainer will add padding it is needed
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False,
    num_proc=4,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
)


gc.collect()
torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=60,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    optim="adamw_8bit",
    report_to="none",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    max_seq_length=max_seq_length,
    packing=True,
    args=training_args,
)

print("Startting training...")
trainer.train()

# Сохранение
print("Saving model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Done! LoRA adapter is saved:", output_dir)