# count chars
# dedup inter - calculate containment
# dedup intra - 5 grams repetition

# count chars
def count_chars_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        char_count = len(text)
        word_count = len(text.split())
    return char_count, word_count

# main function
if __name__ == "__main__":
    # get list of files in ./data directory
    import os
    from pathlib import Path
    from tqdm import tqdm
    data_dir = Path("./data")
    files = list(data_dir.glob("*.txt"))

    files_char_counts = {}
    files_word_counts = {}
    for f_path in tqdm(files):
        char_count, word_count = count_chars_words(f_path)
        files_char_counts[f_path.name] = char_count
        files_word_counts[f_path.name] = word_count
    
    print("Chars |||||||||||||||||||||||||||||")
    print(files_char_counts)
    print("Words |||||||||||||||||||||||||||||")
    print(files_word_counts)
        
