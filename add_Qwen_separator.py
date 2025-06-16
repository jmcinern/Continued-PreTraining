# program to add <|endoftext|> separator at sentence boundary in national corpus data as it has
# - been randomized at sentence level so this will let Qwen dynamically adjust context weighting 
# - during pretraining.

import stanza

SEP = "<|endoftext|>"

# Global variable to store the nlp pipeline to avoid re-initialization
nlp = None

def get_nlp_pipeline():
    """Get or initialize the Stanza pipeline for Irish."""
    global nlp
    if nlp is None:
        try:
            nlp = stanza.Pipeline('ga', verbose=False)
        except:
            # If the model isn't downloaded, download it first
            stanza.download('ga')
            nlp = stanza.Pipeline('ga', verbose=False)
    return nlp

def add_separator(text, separator):
    """
    Add separators at sentence boundaries in Irish text using Stanza.
    
    Args:
        text (str): Input text to process
        separator (str): Separator to add at sentence boundaries
        
    Returns:
        str: Text with separators added at sentence boundaries
    """
    if not text or not text.strip():
        return separator
    
    # Get the nlp pipeline
    nlp_pipeline = get_nlp_pipeline()
    
    # Process the input text
    doc = nlp_pipeline(text)
    sentences = [sentence.text.strip() for sentence in doc.sentences if sentence.text.strip()]    # Add separator at the end of each sentence
    if sentences:
        sentences_with_separator = [sentence + separator for sentence in sentences]
        new_text = "".join(sentences_with_separator)
    else:
        new_text = separator
    
    return new_text

def add_separator_simple(text, separator):
    """
    Simple approach: replace newlines with separators.
    
    Args:
        text (str): Input text to process
        separator (str): Separator to replace newlines with
        
    Returns:
        str: Text with newlines replaced by separators
    """
    if not text or not text.strip():
        return separator
    
    # Replace newlines with the separator
    result = text.replace('\n', separator)
    
    # Ensure it ends with the separator if it doesn't already
    if not result.endswith(separator):
        result += separator
    
    return result

# Legacy code for processing DCU.txt file
if __name__ == "__main__":
    with open("./data/DCU.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    processed_text = add_separator(raw_text, SEP)
    print(f"Processed {len(raw_text)} characters into {len(processed_text)} characters")
    print(f"First 200 characters: {processed_text[:200]}...")
