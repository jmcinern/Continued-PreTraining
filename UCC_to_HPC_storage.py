from datasets import load_dataset

# Load the dataset
ds = load_dataset("ReliableAI/Irish-Text-Collection")

# Save to local disk under ./data
ds.save_to_disk("./data")
