from transformers import AutoTokenizer
import os

def merge_tokenizer_and_push_to_hub(hf_token, tokinizer_name):
    # Original tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B", trust_remote_code=True, use_fast=True
    )

    # Corpus processing
    corpus_path = "./data/NCI_ga.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    

    corpus_iter = corpus_text.split(". ")
    print("NCI |||||||||||||||||||||||||||||||||||||")
    
    print(corpus_iter[-5:])
    with open("./data/UCC_culturax.txt", "r", encoding="utf-8") as f:
        corpus_text = f.read()
    
    
    #corpus_iter += corpus_text.split(". ")
    print("cultura |||||||||||||||||||||||||||||||||||||")
    print(corpus_iter[-5:])  # Print last 5 sentences for debugging
    
    corpus_path = "./data/dail.txt"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()
    
    corpus_iter += corpus_text.split("<|endoftext|>")
    print("dail |||||||||||||||||||||||||||||||||||||")
    print(corpus_iter[-5:])

    with open("./wikipedia_sample.txt", "r", encoding="utf-8") as f:
        corpus_text = f.read()

    corpus_iter += corpus_text.split("\n")
    print("wikipedia |||||||||||||||||||||||||||||||||||||")
    print(corpus_iter[-5:])  # Print last 5 sentences for debugging
    

    with open("./data/DCU_SEP.txt", "r", encoding="utf-8") as f:
        corpus_text += f.read()

    corpus_iter += corpus_text.split("<|endoftext|>")
    print("DCU |||||||||||||||||||||||||||||||||||||")
    print(corpus_iter[-5:])


    # Step 1: Temporary tokenizer for identifying tokens
    temp_tokenizer = base_tokenizer.train_new_from_iterator(
        corpus_iter, vocab_size=30_000
    )

    # Step 2: Identify new tokens so as not to mess up the original KV with duplicates.
    temp_vocab = set(temp_tokenizer.get_vocab().keys())
    base_vocab = set(base_tokenizer.get_vocab().keys())
    new_tokens = list(temp_vocab - base_vocab)


    
    print(f'new tokens: {len(new_tokens)}')
    sample_tokens = [token.replace("Ġ", " ") for token in new_tokens[:25]]
    print(f'sample of new tokens: {sample_tokens}')
    base_tokenizer.add_tokens(new_tokens)

    # Save tokenizer
    temp_tokenizer.save_pretrained(f"./{tokinizer_name}")
    temp_tokenizer.push_to_hub(f"jmcinern/{tokinizer_name}", token=hf_token)

# main
if __name__ == "__main__":
    # ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\
    # Testing
    tokinizer_name = "qwen_tkn_ga_en_big"
    print("Ruinning the tokenizer trainer")
    hf_tkn = os.getenv("HF_KEY")
    merge_tokenizer_and_push_to_hub(hf_tkn, tokinizer_name)
    sample_text = "Chuaigh mé go dtí an siopa agus cheannaigh mé bainne, bhí sé go háilinn. What about you? Did you buy anything from the shop? Lascaine a bhí ann, everything was half price."
    tkn_extended = AutoTokenizer.from_pretrained(f"jmcinern/{tokinizer_name}", trust_remote_code=True)

    tkn_old_model = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    tokenized_sample_old = tkn_old_model(sample_text)
    tokenized_sample_extended = tkn_extended(sample_text)

    # Replace Ġ with spaces for readability
    tkns_old_lst = [token.replace("Ġ", " ") for token in tokenized_sample_old.tokens()]
    tkns_extended_lst = [token.replace("Ġ", " ") for token in tokenized_sample_extended.tokens()]

    print(f'before extending: {len(tkns_old_lst)}, after extending: {len(tkns_extended_lst)}')

    en_test = "this is an english language test, let's see how it works with the extended tokenizer."
    print(f'base: {tkn_old_model(sample_text).tokens()}')
    print(f'extended: {tkn_extended(sample_text).tokens()}')


    # Decode tokens for Qwen readability (just for testoing purposes)
    def decode_qwen_tokenizer_tokens(tokens_lst):
        # Monkey patch fixes for specific Irish characters
        irish_patches = {
            'ÃŃ': 'í',     # ← The fix for your issue!
            'Ã³': 'ó', 'Ã¡': 'á', 'Ã©': 'é', 'Ãº': 'ú',
            'Ã"': 'Ó', 'Ã': 'Á', 'Ã‰': 'É', 'Ãš': 'Ú', 'Ã': 'Í'
        }
        
        decoded_tkns = []
        for tkn in tokens_lst:
            # Apply monkey patches first
            for corrupted, correct in irish_patches.items():
                tkn = tkn.replace(corrupted, correct)
            
            # Then try the general latin-1 fix
            try:
                tkn = tkn.encode('latin-1').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass  # Keep token as-is if decode fails
                
            decoded_tkns.append(tkn)
        return decoded_tkns


    print("TESING ON IRISH")
    print("BEFORE TRAINING")
    print(" | ".join(decode_qwen_tokenizer_tokens(tkns_old_lst)))
    print("AFTER TRAINING")
    print(" | ".join(decode_qwen_tokenizer_tokens(tkns_extended_lst)))
    print (f'\n\n{len(tkns_old_lst)} tokens before extending, {len(tkns_extended_lst)} tokens after extending.\n\n')