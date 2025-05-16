# Overview: practice python script to get familiar with libraries required for continued pre-trainiing
# txt -> tokenizer -> chunking -> trainer (CLM) (with datacollator (for batching)) -> model
# librsaries:
from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
import os
# TXT: raw data

# read in raw text (nce_ga)
with open("./data/nce_ga.txt", "r", encoding="utf-8") as f:
    nec_text = f.read()
    nce_text_words = nec_text.split()
    nce_1M_words = nce_text_words[:10000]  # take first 1M words
    nce_1M_words = " ".join(nce_1M_words)

# read in dáil text
with open("./data/dáil_who_said_what.txt", "r", encoding="utf-8") as f:
    dail_text = f.read()
    dail_text_words = dail_text.split()
    dail_1M_words = dail_text_words[:10000]  # take first 1M words
    dail_1M_words = " ".join(dail_1M_words)

'''
with open("./data/first_1000_words.txt", "w", encoding="utf-8") as f:
    f.write(nce_1M_words[:1000])
    f.write("\n")
    f.write(dail_1M_words[:1000])
'''

# TOKENIZATION
# load in smallest qwen model, practice caching
cache_path = "./cache/qwen3-0.6b"
model_name = "Qwen/Qwen3-0.6B" 
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir=cache_path, 
                                          trust_remote_code=True, #  custom qwen3 code for loading)
)
# tokenize the full texts
# have to break into chunks as model has a max length
nce_raw_chunks = nce_1M_words.split("\n")
dail_raw_chunks = dail_1M_words.split("\n")

# create a dataset from the chunks
nce_dataset = Dataset.from_dict({"text": nce_raw_chunks})
dail_dataset = Dataset.from_dict({"text": dail_raw_chunks})

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
    concatenated = sum(examples["input_ids"], [])
    total = len(concatenated) // block_size * block_size
    input_ids = [concatenated[i:i+block_size] for i in range(0, total, block_size)]
    return {"input_ids": input_ids}

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
    per_device_train_batch_size=2,
    save_steps=50,
    max_steps=300,
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
trainer.save_model("./checkpoints/qwen3-0.6b-CPT_dáil_and_ga")