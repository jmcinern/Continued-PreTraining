import os
import math
import torch
import wandb
import getpass
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# Get wandb API key
wandb_api_key = os.getenv('WANDB_API_KEY')

# Check GPU availability
if torch.cuda.is_available():
    print("CUDA is available!")
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

# Configuration
model_size = "0.6"
model_test_name = f"qwen3-{model_size}B-CPT_ga_NCI_100K_test"

# Load and prepare data (first 10k words from NCI_ga.txt)
print("Loading data...")
with open("./data/NCI_ga.txt", "r", encoding="utf-8") as f:
    nci_all_words = f.read()
    nci_10k = nci_all_words.split()[:100_000]

print(f"Loaded {len(nci_10k):,} words from NCI_ga.txt")

# Chunk data (1000 words per chunk)
chunks_nci = [" ".join(nci_10k[i:i+1000]) 
              for i in range(0, len(nci_10k), 1000)]

print(f"Created {len(chunks_nci)} chunks")

# Create train/val/test splits
chunks_train, chunks_temp = train_test_split(
    chunks_nci, test_size=0.2, random_state=42, shuffle=True
)
chunks_test, chunks_val = train_test_split(
    chunks_temp, test_size=0.5, random_state=42, shuffle=True
)

print(f"Train chunks: {len(chunks_train)}")
print(f"Val chunks: {len(chunks_val)}")
print(f"Test chunks: {len(chunks_test)}")

# Load tokenizer and model
print("Loading tokenizer and model...")
cache_path = f"./cache/qwen3-{model_size}B"
model_name = f"Qwen/Qwen3-{model_size}B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True,
)

# Create dataset
nci_dataset = DatasetDict({
    "train": Dataset.from_dict({"text": chunks_train}),
    "validation": Dataset.from_dict({"text": chunks_val}),
    "test": Dataset.from_dict({"text": chunks_test}),
})

# Tokenization function
def tokenize_function(raw_chunk):
    return tokenizer(raw_chunk['text'])

# Tokenize dataset
print("Tokenizing dataset...")
nci_tokenized_dataset = nci_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"]
)

# Group texts into blocks
block_size = 2048

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total = len(concatenated) // block_size * block_size
    input_chunks = [concatenated[i:i+block_size] for i in range(0, total, block_size)]
    return {"input_ids": input_chunks, "labels": input_chunks}

# Apply grouping
print("Grouping texts into blocks...")
nci_dataset_chunks = nci_tokenized_dataset.map(
    group_texts,
    batched=True,
    remove_columns=["attention_mask"]
)

print(f"Final dataset sizes:")
print(f"  Train: {len(nci_dataset_chunks['train'])}")
print(f"  Val: {len(nci_dataset_chunks['validation'])}")
print(f"  Test: {len(nci_dataset_chunks['test'])}")

# Initialize wandb
print("Initializing wandb...")
wandb.login(key=wandb_api_key)

# wandb config
config = {
    "model_size": f"{model_size}B",
    "model_name": model_name,
    "dataset_name": "NCI_ga_10K",
    "dataset_source": "./data/NCI_ga.txt",
    "total_words": len(nci_10k),
    "total_chunks": len(chunks_nci),
    "train_size": len(chunks_train),
    "val_size": len(chunks_val),
    "test_size": len(chunks_test),
    "train_ratio": 0.9,
    "val_ratio": 0.05,
    "test_ratio": 0.05,
    "chunk_size_words": 1000,
    "block_size": block_size,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "train_batch_size": 16,  # 1 * 8 * 4 GPUs (from deepspeed)
    "learning_rate": 5e-5,
    "epochs": 2,
    "fp16": True,
    "deepspeed_stage": 2,
    "gradient_checkpointing": True,
    "warmup_steps": 100,
    "logging_steps": 50,
    "eval_steps": 100,
    "save_steps": 200,
}

# Start wandb run
wandb.init(
    project="qwen3-irish-cpt",
    name=model_test_name,
    config=config,
    tags=["test-run", "qwen3", "irish", "deepspeed", "multi-gpu"]
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./checkpoints/{model_test_name}",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    do_eval=True,
    eval_strategy="steps",
    prediction_loss_only=True,
    fp16=True,
    dataloader_drop_last=True,
    gradient_checkpointing=True,
    deepspeed="./ds_config.json",
    report_to="wandb",  # Enable wandb logging
    run_name=model_test_name,
    logging_dir=f"./logs/{model_test_name}",
)

# Create trainer
print("Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=nci_dataset_chunks['train'],
    eval_dataset=nci_dataset_chunks['validation'],
    data_collator=data_collator,
)

# Log some sample data to wandb
sample_text = chunks_train[0][:200] + "..."
wandb.log({
    "sample_text": sample_text,
    "vocab_size": tokenizer.vocab_size,
    "model_parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
})

# Start training
print("Starting training...")
try:
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=nci_dataset_chunks['test'])
    
    # Log final metrics
    wandb.log({
        "final_test_loss": test_metrics.get("eval_loss", 0),
        "final_test_perplexity": test_metrics.get("eval_perplexity", 0),
    })
    
    print("Test metrics:", test_metrics)
    
    # Save model
    print("Saving model...")
    trainer.save_model(f"./checkpoints/{model_test_name}")
    
    # Save model artifact to wandb
    model_artifact = wandb.Artifact(
        name=f"{model_test_name}-model",
        type="model",
        description=f"Qwen3-{model_size}B fine-tuned on NCI Irish (10K words)"
    )
    model_artifact.add_dir(f"./checkpoints/{model_test_name}")
    wandb.log_artifact(model_artifact)
    
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Training failed with error: {e}")
    wandb.log({"training_error": str(e)})
    
finally:
    # Finish wandb run
    wandb.finish()

print("Wandb test run completed!")