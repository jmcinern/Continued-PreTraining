import pandas as pd
import os

def txt_word_count_and_sample(f_path):
    wc = 0
    sample = []
    with open(f_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            wc += len(line.split())
            if i < 50:  # collect only the first 50 lines
                sample.append(line.strip())
    return wc, sample


def f_name_from_path(f_path):
    parts = f_path.split("/")
    return parts[-1].split(".")[0]

# get all paths from ./data/ directory
f_paths = []
for root, dirs, files in os.walk("./data/"):
    for file in files:
        if file.endswith(".txt"):
            f_paths.append(os.path.join(root, file))


file_names = []
word_counts = []
samples = []
for f_path in f_paths:
    file_name = f_name_from_path(f_path)
    wc, sample = txt_word_count_and_sample(f_path)
    file_names.append(file_name)
    word_counts.append(wc)
    samples.append(sample)


df = pd.DataFrame({"file_name": file_names, "word_count": word_counts, "first_50_lines": samples})
# Save the DataFrame to a CSV file
df.to_csv("./data/word_counts.csv", index=False)

