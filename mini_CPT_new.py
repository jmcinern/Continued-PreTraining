# Overview: Multi-file concurrent training pipeline for continued pre-training
# Loads all .txt files, creates mixed-batch training with per-file tracking
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, default_data_collator
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
import os
import math
import torch
import wandb
import glob
from collections import defaultdict

class PerFileMetricsCallback(TrainerCallback):
    """Custom callback to track and log per-file metrics during training"""
    
    def __init__(self, file_to_id_mapping):
        super().__init__()
        self.file_to_id_mapping = file_to_id_mapping
        self.id_to_file_mapping = {v: k for k, v in file_to_id_mapping.items()}
        self.step_losses = defaultdict(list)
        self.step_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log per-file metrics to wandb"""
        if logs is None:
            return
            
        # Only process during training
        if state.is_in_train:
            self.step_count += 1
            
            # Log the overall metrics as they are
            for key, value in logs.items():
                if key.startswith('train_'):
                    wandb.log({f"overall/{key}": value}, step=state.global_step)
                    
    def log_per_file_eval_metrics(self, eval_results, step):
        """Log per-file evaluation metrics"""
        for file_prefix, metrics in eval_results.items():
            for metric_name, value in metrics.items():
                wandb.log({f"{file_prefix}/{metric_name}": value}, step=step)

class MultiFileDataCollator:
    """Custom data collator that preserves file_id information for per-file tracking"""
    
    def __init__(self, tokenizer, pad_to_multiple_of=None, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
    def __call__(self, features):
        # Extract file_ids before processing
        file_ids = [f.pop("file_id", -1) for f in features]
        
        # Use default data collator for causal LM
        batch = default_data_collator(features)
        
        # Add file_ids back to the batch
        batch["file_ids"] = torch.tensor(file_ids, dtype=torch.long)
        
        return batch

def discover_and_load_files(data_dir="./data", max_words=100000):
    """Discover all .txt files and load first max_words from each"""
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    file_contents = {}
    
    if not txt_files:
        print(f"Warning: No .txt files found in {data_dir}")
        return file_contents
    
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Split into words and take first max_words (not chars!)
                all_words = content.split()
                if len(all_words) < max_words:
                    print(f"Warning: {filename} has only {len(all_words)} words, using all available")
                    words_subset = all_words
                else:
                    words_subset = all_words[:max_words]
                
                # Rejoin words back to text
                content_subset = " ".join(words_subset)
                
                if len(words_subset) < 1000:  # Skip very small files
                    print(f"Skipping {filename}: too small ({len(words_subset)} words)")
                    continue
                    
                file_contents[filename] = content_subset
                print(f"Loaded {len(words_subset)} words from {filename}")
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return file_contents

def prepare_bible_ood(bible_content, ood_words=10000):
    """Prepare bible content for OOD validation/test"""
    # Take first ood_words words (not chars!)
    all_words = bible_content.split()
    ood_words_subset = all_words[:ood_words]
    
    # Split into validation and test (50/50)
    mid_point = len(ood_words_subset) // 2
    
    val_text = " ".join(ood_words_subset[:mid_point])
    test_text = " ".join(ood_words_subset[mid_point:])
    
    return val_text, test_text

def chunk_and_process_file(content, filename, chunk_size=1000, file_id=0):
    """Process a single file: chunk, tokenize, and add file_id"""
    words = content.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    # Split into train/val/test (94/3/3)
    train_chunks, temp_chunks = train_test_split(chunks, test_size=0.06, random_state=42, shuffle=True)
    val_chunks, test_chunks = train_test_split(temp_chunks, test_size=0.5, random_state=42, shuffle=True)
    
    return {
        'train': train_chunks,
        'val': val_chunks,
        'test': test_chunks,
        'filename': filename,
        'file_id': file_id
    }

def tokenize_function(examples, tokenizer):
    """Tokenize text chunks"""
    return tokenizer(examples['text'])

def group_texts(examples, block_size=2048):
    """Group texts into blocks of specified size"""
    concatenated = sum(examples["input_ids"], [])
    total = len(concatenated) // block_size * block_size
    input_chunks = [concatenated[i:i+block_size] for i in range(0, total, block_size)]
    
    # Preserve file_ids for each chunk
    file_ids = examples.get("file_id", [])
    if file_ids:
        # Repeat file_id for each chunk created from this batch
        chunk_file_ids = []
        for file_id in file_ids:
            chunk_file_ids.extend([file_id] * len(input_chunks))
        return {"input_ids": input_chunks, "labels": input_chunks, "file_id": chunk_file_ids[:len(input_chunks)]}
    else:
        return {"input_ids": input_chunks, "labels": input_chunks}

def main():
    # Check if this is the main process in distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    
    # Configuration
    model_size = "0.6"
    model_test_name = f"MULTI-FILE-DS-{model_size}B-CPT_ga_wandb"
    LR = 1e-4
    block_size = 2048
    chunk_size = 1000
    
    # Initialize wandb only on main process
    if is_main_process:
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
    
    config = {
        "model_size": f"{model_size}B",
        "epochs": 2,        "learning_rate": LR,
        "block_size": block_size,
        "chunk_size": chunk_size,
        "training_approach": "multi-file-concurrent"
    }
    
    if is_main_process:
        wandb.init(
            project="qwen3-irish-cpt-multifile",
            name=model_test_name,
            config=config,
            tags=["multi-file", "qwen3", "irish", "deepspeed", "mixed-batch"]
        )
    
    print("CUDA availability:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Step 1: Discover and load all files
    print("\n=== Step 1: Loading files ===")
    file_contents = discover_and_load_files("./data", max_words=100000)
    
    # Step 2: Handle bible_gaeilge.txt specially for OOD
    bible_val_text, bible_test_text = None, None
    if "bible_gaeilge.txt" in file_contents:
        bible_val_text, bible_test_text = prepare_bible_ood(file_contents["bible_gaeilge.txt"])
        del file_contents["bible_gaeilge.txt"]  # Remove from training files
        print("Prepared bible_gaeilge.txt for OOD evaluation")
    
    # Step 3: Create file ordering (bitext first, then alphabetical)
    training_files = list(file_contents.keys())
    bitext_files = [f for f in training_files if "bitext" in f.lower()]
    other_files = sorted([f for f in training_files if "bitext" not in f.lower()])
    ordered_files = bitext_files + other_files
    
    print(f"\nTraining file order: {ordered_files}")
    
    # Step 4: Create file-to-ID mapping
    file_to_id = {filename: idx for idx, filename in enumerate(ordered_files)}
    id_to_file = {idx: filename for filename, idx in file_to_id.items()}
    
    # Step 5: Process each file
    print("\n=== Step 2: Processing files ===")
    all_processed_data = {}
    
    for filename in ordered_files:
        content = file_contents[filename]
        file_id = file_to_id[filename]
        processed = chunk_and_process_file(content, filename, chunk_size, file_id)
        all_processed_data[filename] = processed
        print(f"Processed {filename}: {len(processed['train'])} train, {len(processed['val'])} val, {len(processed['test'])} test chunks")
      # Step 6: Load tokenizer and model
    print("\n=== Step 3: Loading model and tokenizer ===")
    cache_path = f"./cache/qwen3-{model_size}B"
    model_name = f"Qwen/Qwen3-{model_size}B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_path, 
        trust_remote_code=True
    )
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_path,
        trust_remote_code=True
    )
    
    print(f"Model loaded: {model_name}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {model.num_parameters():,}")
      # Step 7: Skip mixed dataset creation, process files individually
    print("\n=== Step 4: Ready for file-by-file processing ===")
    print(f"Will process {len(ordered_files)} files in order: {ordered_files}")
    
    # Step 8: Tokenize and group by file first to ensure purity
    print("\n=== Step 5: Tokenizing and grouping by file ===")
    
    def tokenize_with_file_id(examples):
        tokenized = tokenizer(examples['text'])
        tokenized['file_id'] = examples['file_id']
        return tokenized
    
    def group_texts_with_file_id(examples):
        # This function processes examples that should already be from the same file
        concatenated = sum(examples["input_ids"], [])
        total = len(concatenated) // block_size * block_size
        input_chunks = [concatenated[i:i+block_size] for i in range(0, total, block_size)]
        
        # Since all examples in this batch should have the same file_id, use the first one
        file_ids = examples["file_id"]
        if len(input_chunks) > 0:
            # All chunks from this batch get the same file_id (the first one)
            chunk_file_ids = [file_ids[0]] * len(input_chunks)
        else:
            chunk_file_ids = []
        
        return {
            "input_ids": input_chunks, 
            "labels": input_chunks,
            "file_id": chunk_file_ids
        }
    
    def tokenize_and_group_by_file(chunks, file_id):
        """Tokenize and group chunks from a single file"""
        if not chunks:
            return []
            
        # Create dataset for this file only
        file_dataset = Dataset.from_dict({
            "text": chunks,
            "file_id": [file_id] * len(chunks)
        })
        
        # Tokenize
        file_tokenized = file_dataset.map(tokenize_with_file_id, batched=True, remove_columns=["text"])
        
        # Group into blocks
        file_grouped = file_tokenized.map(group_texts_with_file_id, batched=True, remove_columns=["attention_mask"])
        
        return file_grouped
    
    # Process each file separately to maintain purity
    all_train_grouped = []
    all_val_grouped = []
    all_test_grouped = []
    
    for filename, data in all_processed_data.items():
        file_id = file_to_id[filename]
        
        train_grouped = tokenize_and_group_by_file(data['train'], file_id)
        val_grouped = tokenize_and_group_by_file(data['val'], file_id)
        test_grouped = tokenize_and_group_by_file(data['test'], file_id)
        
        if len(train_grouped) > 0:
            all_train_grouped.append(train_grouped)
        if len(val_grouped) > 0:
            all_val_grouped.append(val_grouped)
        if len(test_grouped) > 0:
            all_test_grouped.append(test_grouped)
            
        print(f"Processed {filename}: {len(train_grouped)} train, {len(val_grouped)} val, {len(test_grouped)} test blocks")
    
    # Concatenate all file datasets
    if all_train_grouped:
        train_chunks = concatenate_datasets(all_train_grouped).shuffle(seed=42)
    else:
        train_chunks = Dataset.from_dict({"input_ids": [], "labels": [], "file_id": []})
        
    if all_val_grouped:
        val_chunks = concatenate_datasets(all_val_grouped)
    else:
        val_chunks = Dataset.from_dict({"input_ids": [], "labels": [], "file_id": []})
        
    if all_test_grouped:
        test_chunks = concatenate_datasets(all_test_grouped)
    else:
        test_chunks = Dataset.from_dict({"input_ids": [], "labels": [], "file_id": []})
    
    print(f"Final mixed datasets: {len(train_chunks)} train, {len(val_chunks)} val, {len(test_chunks)} test blocks")
    
    # Step 10: Setup training
    print("\n=== Step 7: Setting up training ===")
      # Custom data collator and callback
    data_collator = MultiFileDataCollator(tokenizer=tokenizer)
    metrics_callback = PerFileMetricsCallback(file_to_id)
    
    training_args = TrainingArguments(
        learning_rate=LR,
        output_dir=f"./checkpoints/{model_test_name}",
        overwrite_output_dir=True,
        num_train_epochs=2,
        save_steps=500,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=3,
        do_eval=True,
        eval_steps=3,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        report_to="wandb",
        deepspeed="./ds_config.json",
        gradient_checkpointing=True,
    )
    
    def compute_metrics(eval_preds):
        loss = eval_preds.metrics.get("eval_loss", 0)
        return {"perplexity": math.exp(loss)}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_chunks,
        eval_dataset=val_chunks,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]
    )
    
    print(f"Train batches per epoch: {len(train_chunks) // training_args.per_device_train_batch_size}")
    
    # Step 11: Train
    print("\n=== Step 8: Training ===")
    trainer.train()
    
    # Step 12: Evaluation on test sets
    print("\n=== Step 9: Final evaluation ===")
    
    # Evaluate on overall test set
    test_metrics = trainer.evaluate(eval_dataset=test_chunks)
    if test_metrics:
        wandb.log({
            "final_test_loss": test_metrics.get("eval_loss"),
            "final_test_perplexity": test_metrics.get("eval_perplexity", math.exp(test_metrics.get("eval_loss", 0))),
        })
    
    # Per-file evaluation
    eval_results = {}
    
    for filename, data in all_processed_data.items():
        file_id = file_to_id[filename]
        
        # Create test dataset for this file only
        file_test_chunks = [{"text": chunk, "file_id": file_id} for chunk in data['test']]
        if file_test_chunks:
            file_test_dataset = Dataset.from_list(file_test_chunks)
            file_test_tokenized = file_test_dataset.map(tokenize_with_file_id, batched=True, remove_columns=["text"])
            file_test_grouped = file_test_tokenized.map(group_texts_with_file_id, batched=True, remove_columns=["attention_mask"])
            
            if len(file_test_grouped) > 0:
                file_metrics = trainer.evaluate(eval_dataset=file_test_grouped)
                eval_results[f"file_{filename.replace('.txt', '')}"] = {
                    "eval_loss": file_metrics.get("eval_loss"),
                    "eval_perplexity": file_metrics.get("eval_perplexity", math.exp(file_metrics.get("eval_loss", 0)))
                }
      # Bible OOD evaluation
    if bible_val_text and bible_test_text:
        # Use a special file_id for bible data (e.g., -1)
        bible_file_id = -1
        
        for bible_split, bible_text in [("bible_val", bible_val_text), ("bible_test", bible_test_text)]:
            bible_chunks = [" ".join(bible_text.split()[i:i+chunk_size]) 
                          for i in range(0, len(bible_text.split()), chunk_size)]
            if bible_chunks:
                # Create dataset with file_id
                bible_dataset = Dataset.from_dict({
                    "text": bible_chunks,
                    "file_id": [bible_file_id] * len(bible_chunks)
                })
                bible_tokenized = bible_dataset.map(tokenize_with_file_id, batched=True, remove_columns=["text"])
                bible_grouped = bible_tokenized.map(group_texts_with_file_id, batched=True, remove_columns=["attention_mask"])
                
                if len(bible_grouped) > 0:
                    bible_metrics = trainer.evaluate(eval_dataset=bible_grouped)
                    eval_results[f"ood_{bible_split}"] = {
                        "eval_loss": bible_metrics.get("eval_loss"),
                        "eval_perplexity": bible_metrics.get("eval_perplexity", math.exp(bible_metrics.get("eval_loss", 0)))
                    }
    
    # Log all evaluation results
    metrics_callback.log_per_file_eval_metrics(eval_results, trainer.state.global_step)
    
    # Step 13: Save model
    print("\n=== Step 10: Saving model ===")
    trainer.save_model(f"./checkpoints/{model_test_name}")
    
    wandb.finish()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
