import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from auto_retrain import get_next_model_version

# -----------------------------
# CONFIG
# -----------------------------

BASE_MODEL = "google/flan-t5-small"
DATA_PATH = "training/train.json"

version = get_next_model_version()

OUTPUT_DIR = f"models/orchestrator_v{version}"

print("Saving model version:", version)

# -----------------------------
# LOAD TOKENIZER
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# -----------------------------
# LOAD DATASET
# -----------------------------

dataset = load_dataset(
    "json",
    data_files=DATA_PATH
)["train"]

# -----------------------------
# PREPROCESS
# -----------------------------

def preprocess(example):

    instruction = example.get("instruction") or example.get("input") or ""
    output = example.get("output") or ""

    model_input = f"""
Instruction:
{instruction}
"""

    inputs = tokenizer(
        model_input,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    targets = tokenizer(
        output,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    inputs["labels"] = targets["input_ids"]

    return inputs

dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)

dataset.set_format("torch")

# -----------------------------
# LOAD MODEL
# -----------------------------

model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

model.to("cpu")

# -----------------------------
# LORA CONFIG
# -----------------------------

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=5,
    save_strategy="epoch",
    remove_unused_columns=False,
    dataloader_pin_memory=False
)

# -----------------------------
# TRAINER
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# -----------------------------
# TRAIN
# -----------------------------

trainer.train()

# -----------------------------
# SAVE
# -----------------------------

model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("LoRA training complete.")