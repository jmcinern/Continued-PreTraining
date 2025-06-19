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
model_test_name = "NO_DS-"+model_size+"B-CPT_ga_wandb_tests"

wandb_api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_api_key)
import wandb
import time

wandb.init(project="test-logging")

for i in range(10):
    wandb.log({"test_loss": 1.0 - i*0.1, "step": i})
    time.sleep(2)
    print(f"Logged step {i}")

wandb.finish()
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

# agent: eval "$(ssh-agent -s)"
# ssh-add ~/.ssh/id_ed25519_personal
# TXT: raw data

# read in raw text (nce_ga)
with open("./data/NCI_ga.txt", "r", encoding="utf-8") as f:
    nce_all_words = f.read()
    nce_1M = nce_all_words.split()[:100_000]
    
# read in dáil text
with open("./data/dáil_who_said_what.txt", "r", encoding="utf-8") as f:
    dail_all_words = f.read()
    dail_1M = dail_all_words.split()[:100_000]

# chunk before tokenization
chunks_nce = [" ".join(nce_1M[i:i+1000])
          for i in range(0, len(nce_1M), 1000)]

# train set - 94%
chunks_nce_train, chunks_nce_06_tmp = train_test_split(chunks_nce,
    test_size=0.06, random_state=42, shuffle=True)

# test and val set - 3% each 
chunks_nce_test, chunks_nce_val = train_test_split(chunks_nce_06_tmp,
    test_size=0.5, random_state=42, shuffle=True)


chunks_dail = [" ".join(dail_1M[i:i+1000]) 
          for i in range(0, len(dail_1M), 1000)] 
# TOKENIZATION
# load in smallest qwen model, practice caching
cache_path = "./cache/qwen3-"+model_size+"B"
model_name = "Qwen/Qwen3-"+model_size+"B" 
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          cache_dir=cache_path, 
                                          trust_remote_code=True, #  custom qwen3 code for loading)
)

# create a dataset
nce_dataset = DatasetDict({
    "train": Dataset.from_dict({"text": chunks_nce_train}),
    "validation": Dataset.from_dict({"text": chunks_nce_val}),
    "test": Dataset.from_dict({"text": chunks_nce_test}),
})
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
nce_dataset_chunks = nce_tokenized_dataset.map(group_texts, 
                                                    batched=True, 
                                                    # attn padding not important for CPT
                                                    remove_columns=["attention_mask"] 
                                                    )
'''dail_dataset_chunks = dail_tokenized_dataset.map(group_texts, 
                                                      batched=True,
                                                      remove_columns=["attention_mask"]
                                                      )
                                                      '''

# now we just have input_ids: tokens (represented by numbers)

'''
# mix the datasets
mixed_dataset = concatenate_datasets([
    nce_dataset_chunks,
    dail_dataset_chunks
]).shuffle(seed=42)
# now load base model
'''



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
    logging_steps=10,
    do_eval= True,
    eval_steps=10,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,
    report_to="wandb",  # enable wandb/hub
    #deepspeed="./ds_config.json", # deepspeed config
    gradient_checkpointing=True, # trick to save subsection of forward pass, prevents caching if True.
)

# PPL
def compute_metrics(eval_preds):
    loss = eval_preds.metrics["eval_loss"]
    return {"perplexity": math.exp(loss)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=nce_dataset_chunks['train'],
    eval_dataset=nce_dataset_chunks['validation'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

print(f"Train batches per epoch: {len(nce_dataset_chunks['train']) // (training_args.per_device_train_batch_size)}")

trainer.train()

# evaluate on the test set
test_metrics = trainer.evaluate(eval_dataset=nce_dataset_chunks['test'])

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
