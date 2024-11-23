import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

def prepare_dataset(file_path):
    """
    Load and split the dataset into training and validation sets (80% train, 20% eval).
    """
    dataset = load_dataset("json", data_files=file_path)["train"]
    train_size = int(0.8 * len(dataset))  # 80% train
    eval_size = len(dataset) - train_size  # 20% eval
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    return train_dataset, eval_dataset

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenize both the question and generated answer.
    """
    questions = examples["question"]
    answers = examples["generated_answer"]

    inputs = tokenizer(
        questions,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    outputs = tokenizer(
        answers,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs

def main():
    model_name_or_path = "/home/asperger/DPSD/models/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load and split dataset
    dataset_path = "data/gsm8k_generated.jsonl"
    train_dataset, eval_dataset = prepare_dataset(dataset_path)

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["question", "generated_answer"],
    )
    tokenized_eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["question", "generated_answer"],
    )

    training_args = TrainingArguments(
        output_dir="./qwen2.5_0.5B_lora_gsm8k",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        logging_dir="./logs",
        logging_steps=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        gradient_accumulation_steps=32,
        learning_rate=2e-5,
        fp16=True,
        load_best_model_at_end=True,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,  # Pass eval dataset
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./qwen2.5_0.5B_lora_gsm8k")

if __name__ == "__main__":
    main()