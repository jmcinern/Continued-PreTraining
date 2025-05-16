import chromadb
import pandas as pd
import re
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm
import os

tqdm.pandas()

class Dail_dataset:
    def __init__(self, db_fpath, batch_size=10000):
        self.db_fpath = db_fpath
        hf_dataset_path = os.path.join(db_fpath, "data_set_hf")

        if os.path.exists(hf_dataset_path):
            print("Loading dataset from disk...")
            self.dataset = load_from_disk(hf_dataset_path)
            print("Loaded dataset from disk")
            # get the the who said what column, concatenate the text and store as text file
            who_said_what_list = self.dataset["speaker_text"]
            who_said_what = " ".join(who_said_what_list)
            if not os.path.exists("./dáil_who_said_what.txt"):
                print("Creating dáil_who_said_what.txt file...")
                with open("dáil_who_said_what.txt", "w", encoding="utf-8") as f:
                    f.write(who_said_what)
            return
        else:
            print("Dataset not found, creating new dataset in batches...")

            chroma_client = chromadb.PersistentClient(path=db_fpath)
            collection = chroma_client.get_or_create_collection("oireachtas_debates")

            metadata = collection.get(include=["metadatas"])["metadatas"]
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            datasets_batches = []

            for i in tqdm(range(0, len(metadata), batch_size), desc="Processing batches"):
                batch_metadata = metadata[i:i+batch_size]

                df = pd.DataFrame(batch_metadata)
                df["date"] = df["url"].apply(lambda x: re.search(r'/dail/(\d{4}-\d{2}-\d{2})/', x).group(1))
                df["word_count"] = df["text"].apply(lambda x: len(x.split()))
                df["token_count"] = df["text"].apply(lambda x: len(tokenizer.tokenize(x)))
                df["speaker_text"] = df.apply(lambda x: f"{x['speaker']} said: {x['text']}", axis=1)

                batch_dataset = Dataset.from_pandas(df)
                batch_path = os.path.join(db_fpath, f"batch_{i//batch_size}")
                batch_dataset.save_to_disk(batch_path)

                datasets_batches.append(batch_dataset)
                
                del df

            self.dataset = concatenate_datasets(datasets_batches)
            self.dataset.save_to_disk(hf_dataset_path)
            print("Saved full dataset to disk")