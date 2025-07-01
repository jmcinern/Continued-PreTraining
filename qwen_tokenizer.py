from transformers import AutoTokenizer
import os

# --- FIX: Create a generator to iterate over files without loading all to memory ---
def get_corpus_iterator(file_paths, split_on):
    """
    A generator that yields lines from a list of files.
    This avoids loading the entire corpus into memory.
    """
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found, skipping: {file_path}")
            continue
        
        print(f"📖 Processing file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            # Read the file in chunks to handle very large files
            buffer = ""
            while True:
                chunk = f.read(1024 * 1024) # Read 1MB at a time
                if not chunk:
                    if buffer:
                        yield buffer
                    break
                
                buffer += chunk
                while split_on in buffer:
                    line, _, buffer = buffer.partition(split_on)
                    if line.strip(): # Avoid empty lines
                        yield line


def merge_tokenizer_and_push_to_hub(hf_key):
    # Original tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", trust_remote_code=True, use_fast=True
    )

    # List of files to process
    corpus_files = [
        "./data/NCI_ga.txt",
        "./data/dail.txt",
        "./wikipedia_sample.txt",
        "./data/DCU_SEP.txt", 
        "./data/UCC_culturax.txt"
    ]
    
    # Create the memory-efficient iterator
    corpus_iter = get_corpus_iterator(corpus_files, split_on="\n")

    print("\n🚀 Starting tokenizer training from iterator...")
    # Step 1: Train the new tokenizer from the iterator
    # This will now process the files chunk by chunk
    temp_tokenizer = base_tokenizer.train_new_from_iterator(
        corpus_iter, vocab_size=25_000
    )
    print("✅ Tokenizer training complete.")

    # Step 2: Identify new tokens so as not to mess up the original KV with duplicates.
    print("🔍 Identifying new tokens...")
    temp_vocab = set(temp_tokenizer.get_vocab().keys())
    base_vocab = set(base_tokenizer.get_vocab().keys())
    new_tokens = list(temp_vocab - base_vocab)
    
    print(f'Found {len(new_tokens)} new tokens.')
    sample_tokens = [token.replace("Ġ", " ") for token in new_tokens[:25]]
    print(f'Sample of new tokens: {sample_tokens}')
    
    print("➕ Adding new tokens to base tokenizer...")
    base_tokenizer.add_tokens(new_tokens)

    # Save tokenizer
    print("💾 Saving tokenizer locally...")
    output_dir = "./qwen_en_ga_from_scratch2"
    base_tokenizer.save_pretrained(output_dir)
    
    print(f"🚀 Pushing tokenizer to hub: jmcinern/{os.path.basename(output_dir)}")
    base_tokenizer.push_to_hub(f"jmcinern/Qwen_Tokenizer_Ga_En_Big", token=hf_key)
    
    return output_dir


# main
if __name__ == "__main__":
    hf_key = os.getenv("HF_KEY")
    # Create the tokenizer
    local_tokenizer_path = merge_tokenizer_and_push_to_hub(hf_key)

    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
    # Testing
    print("\n\n--- Running Tests ---")
    sample_text = "Chuaigh mé go dtí an siopa agus cheannaigh mé bainne, bhí sé go háilinn. What about you? Did you buy anything from the shop? Lascaine a bhí ann, everything was half price."
    
    # Load the tokenizer you just created locally
    tkn_extended = AutoTokenizer.from_pretrained(local_tokenizer_path, trust_remote_code=True)
    tkn_old_model = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    # Use convert_ids_to_tokens instead of the deprecated .tokens()
    tkns_old_ids = tkn_old_model.encode(sample_text)
    tkns_extended_ids = tkn_extended.encode(sample_text)

    tkns_old_lst = tkn_old_model.convert_ids_to_tokens(tkns_old_ids)
    tkns_extended_lst = tkn_extended.convert_ids_to_tokens(tkns_extended_ids)

    print(f'Tokens before extending: {len(tkns_old_lst)}, Tokens after extending: {len(tkns_extended_lst)}')

    def decode_qwen_tokenizer_tokens(tokens_lst):
        # This function seems fine for debugging display
        irish_patches = {'ÃŃ': 'í', 'Ã³': 'ó', 'Ã¡': 'á', 'Ã©': 'é', 'Ãº': 'ú', 'Ã"': 'Ó', 'Ã': 'Á', 'Ã‰': 'É', 'Ãš': 'Ú', 'Ã': 'Í'}
        decoded_tkns = []
        for tkn in tokens_lst:
            for corrupted, correct in irish_patches.items():
                tkn = tkn.replace(corrupted, correct)
            try:
                tkn = tkn.encode('latin-1').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            decoded_tkns.append(tkn.replace("Ġ", " "))
        return decoded_tkns

    print("\n--- IRISH/ENGLISH TEST ---")
    print("BEFORE TRAINING:")
    print(" | ".join(decode_qwen_tokenizer_tokens(tkns_old_lst)))
    print("\nAFTER TRAINING:")
    print(" | ".join(decode_qwen_tokenizer_tokens(tkns_extended_lst)))
    print(f'\nImprovement: From {len(tkns_old_lst)} tokens down to {len(tkns_extended_lst)} tokens.')