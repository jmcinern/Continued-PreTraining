# Overview: practice python script to get familiar with libraries required for continued pre-trainiing
# txt -> tokenizer -> chunking -> trainer (CLM) (with datacollator (for batching)) -> model
# librsaries:
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict #concatenate_datasets
#import torch
from sklearn.model_selection import train_test_split
import os
import math
import torch
import wandb


model_size = "0.6"
model_test_name = "FRESH_SCRIPT-"+model_size+"B-CPT_ga_wandb_tests"
cache_path = "./cache/qwen3-"+model_size+"B"
model_name = "Qwen/Qwen3-"+model_size+"B"

tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                        cache_dir=cache_path, 
                                        trust_remote_code=True, #  custom qwen3 code for loading)
)

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
import wandb
import time


LR = 1e-4
config = {
    "model_size": f"{model_size}B",
    "epochs": 2,
    "learning_rate": LR,  # or whatever you're using
}
wandb.init(
    project="qwen3-irish-cpt",
    name=model_test_name,
    config=config,
    tags=["test-run", "qwen3", "irish", "deepspeed", "multi-gpu"]
)

if torch.cuda.is_available():
    print("CUDA is available!")
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

# 1. read file
# 2. chunk
# 3. split
# 4. -> dataset
# 5. tokenize
def file_to_chunks(file_path, chunk_size=1000):
    # 1. read file
    with open(file_path, "r", encoding="utf-8") as f:
        file_text = f.read()
        file_words = file_text.split()[:1_000_000]
        
    # 2. chunk
    chunks = [" ".join(file_words[i:i+chunk_size])
            for i in range(0, len(file_words), chunk_size)]
    
    # 3. split
    train, tmp = train_test_split(chunks, test_size=0.06, random_state=42, shuffle=True)

    # test and val set - 3% each 
    test, val = train_test_split(tmp, test_size=0.5, random_state=42, shuffle=True)
    
    # 4. -> dataset
    dataset = DatasetDict({
    "train": Dataset.from_dict({"text": train}),
    "validation": Dataset.from_dict({"text": val}),
    "test": Dataset.from_dict({"text": test}),
    })

    # 5. tokenize
    # b) helper function
    def tokenize_function(raw_chunk):
        return tokenizer(raw_chunk['text'])
    
    # c) tokenize dataset 
    dataset_tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # tokenized -> model input size blocks
    block_size = 2048 

    # turns batch into chunks of block_size
    def group_texts(examples):
        # convert list of lists into a single list
        concatenated = sum(examples["input_ids"], [])
        # calculate max number of tokens given block size.
        total = len(concatenated) // block_size * block_size
        # cut up list by block size
        input_chunks = [concatenated[i:i+block_size] for i in range(0, total, block_size)]
        # need to have labels for the dataset batching 
        return {"input_ids": input_chunks, "labels": input_chunks} 

    
    # apply the function to the tokenized dataset
    dataset_chunks = dataset_tokenized.map(group_texts, 
                                                        batched=True, 
                                                        # attn padding not important for CPT
                                                        remove_columns=["attention_mask"] 
                                                        )
    return dataset_chunks

dataset_chunks = file_to_chunks("./data/NCI_ga.txt", chunk_size=10000)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_path,
    trust_remote_code=True, #  custom qwen3 code for loading
)


# set up trainer with data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # CLM (autoregressive) 
)

question_qualitative = "Inis dom gearrscéal"
# set up training arguments
training_args = TrainingArguments(
    learning_rate=LR,
    output_dir="./checkpoints/"+model_test_name,
    overwrite_output_dir=True,
    num_train_epochs=2,
    save_steps=500,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,#gradient_checkpointing=True, # trick to save subsection of forward pass, prevents caching if True.
    logging_steps=3,
    do_eval= True,
    eval_strategy="steps",
    eval_steps=3,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
    report_to="wandb",  # enable wandb/hub
    deepspeed="./ds_config.json", # deepspeed config
    gradient_checkpointing=True, # trick to save subsection of forward pass, prevents caching if True.
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_chunks['train'],
    eval_dataset=dataset_chunks['validation'],
    data_collator=data_collator,
    )

print(f"Train batches per epoch: {len(dataset_chunks['train']) // (training_args.per_device_train_batch_size)}")

trainer.train()

# evaluate on the test set
test_metrics = trainer.evaluate(eval_dataset=dataset_chunks['test'])

if test_metrics:
    wandb.log({
        "final_test_loss": test_metrics.get("eval_loss"),
        "final_test_perplexity": test_metrics.get("eval_perplexity", math.exp(test_metrics.get("eval_loss", 0))),
    })
wandb.finish()
'''
# then English
trainer.train_dataset = dail_dataset_20.6_chunks
trainer.train(resume_from_checkpoint="./checkpoints/after_irish")
'''
# save the model
trainer.save_model("./checkpoints/"+model_test_name)
