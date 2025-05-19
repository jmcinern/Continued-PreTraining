# Overview: practice python script to get familiar with libraries required for continued pre-trainiing
# txt -> tokenizer -> chunking -> trainer (CLM) (with datacollator (for batching)) -> model
# librsaries:
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import os
# TXT: raw data

# read in raw text (nce_ga)
with open("./data/nce_ga.txt", "r", encoding="utf-8") as f:
    nce_lines = f.readlines()
    

# read in dáil text
with open("./data/dáil_who_said_what.txt", "r", encoding="utf-8") as f:
    dail_lines = f.readlines()

# create helper to limit word count for experimentation
def limit_words(lines, max_words):
    word_count = 0
    limited = []
    for line in lines:
        words = line.split()
        if word_count + len(words) > max_words:
            remaining = max_words - word_count
            limited.append(" ".join(words[:remaining]))
            break
        else:
            limited.append(line)
            word_count += len(words)
    return limited

nce_lines = limit_words(nce_lines, 1_000_000)
dail_lines = limit_words(dail_lines, 1_000_000)


    

'''
with open("./data/first_1000_words.txt", "w", encoding="utf-8") as f:
    f.write(nce_1M_words[:1000])
    f.write("\n")
    f.write(dail_1M_words[:1000])
'''

# TOKENIZATION
# load in smallest qwen model, practice caching
cache_path = "./cache/qwen3-4b"
model_name = "Qwen/Qwen3-4B" 
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir=cache_path, 
                                          trust_remote_code=True, #  custom qwen3 code for loading)
)

# create a dataset from the chunks
nce_dataset = Dataset.from_dict({"text": nce_lines})
dail_dataset = Dataset.from_dict({"text": dail_lines})

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
    logging_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=False,  # use True if on GPU with float16 support
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
