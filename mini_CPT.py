# Overview: practice python script to get familiar with libraries required for continued pre-trainiing
# txt -> tokenizer -> chunking -> trainer (CLM) (with datacollator (for batching)) -> model
# librsaries:
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import os
# TXT: raw data

# read in raw text (nce_ga)
with open("./data/nce_ga.txt", "r", encoding="utf-8") as f:
    nce_all_words = f.read()
    nce_1M = nce_all_words.split()[:1_000_000]
    
# read in dáil text
with open("./data/dáil_who_said_what.txt", "r", encoding="utf-8") as f:
    dail_all_words = f.read()
    dail_1M = dail_all_words.split()[:1_000_000]

# chunk before tokenization
chunks_nce = [" ".join(nce_1M[i:i+1000])
          for i in range(0, len(nce_1M), 1000)]

chunks_dail = [" ".join(dail_1M[i:i+1000]) 
          for i in range(0, len(dail_1M), 1000)] 
# TOKENIZATION
# load in smallest qwen model, practice caching
cache_path = "./cache/qwen3-4b"
model_name = "Qwen/Qwen3-4B" 
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir=cache_path, 
                                          trust_remote_code=True, #  custom qwen3 code for loading)
)

# create a dataset
nce_dataset = Dataset.from_dict({"text": chunks_nce})
dail_dataset = Dataset.from_dict({"text": chunks_dail})


# simple helper function to tokenize the dataset
def tokenize_function(raw_chunk):
    return tokenizer(raw_chunk['text'])


# tokenize after having split into chunks on the new line
nce_tokenized_dataset = nce_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
dail_tokenized_dataset = dail_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# now slice up into blocks to feed into the model
block_size = 2048 # training example size

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
nce_dataset_2048_chunks = nce_tokenized_dataset.map(group_texts, 
                                                    batched=True, 
                                                    # attn padding not important for CPT
                                                    remove_columns=["attention_mask"] 
                                                    )
dail_dataset_2048_chunks = dail_tokenized_dataset.map(group_texts, 
                                                      batched=True,
                                                      remove_columns=["attention_mask"]
                                                      )

# now we just have input_ids: tokens (represented by numbers)

# mix the datasets
mixed_dataset = concatenate_datasets([
    nce_dataset_2048_chunks,
    dail_dataset_2048_chunks
]).shuffle(seed=42)
# now load base model

print(len(mixed_dataset))
print(len(mixed_dataset[0]["input_ids"]))
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto", 
    cache_dir=cache_path,
    trust_remote_code=True, #  custom qwen3 code for loading
)

# set up trainer with data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # CLM (autoregressive) 
)

# set up training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=500,
    gradient_accumulation_steps=16, # smaller gradient updating, after 100 steps not whole batch.
    gradient_checkpointing=True, # trick to save subsection of forward pass
    logging_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # use True if on GPU with float16 support
    report_to="none"  # disable wandb/hub
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mixed_dataset,
    data_collator=data_collator,
)

# train the model
trainer.train()
# save the model
trainer.save_model("./checkpoints/qwen3-4b-CPT_dáil_and_ga")
