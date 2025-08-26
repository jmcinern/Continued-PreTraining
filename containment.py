from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
from collections import defaultdict
import hashlib
import time
N_GRAM_SIZE=10

N_GRAM_SIZE=10

def calculate_ngrams(text):
    """Calculate n-grams from text using ultra-fast method"""
    n_grams = set()
    tokens = text.split()
    
    for i in range(len(tokens) - N_GRAM_SIZE + 1):
        # Hash the tuple directly - no string operations!
        ngram_tuple = tuple(tokens[i:i+N_GRAM_SIZE])
        ngram_hash = hash(ngram_tuple) % (10**8)
        n_grams.add(ngram_hash)
    
    return n_grams


def batch_and_multithread_process_text(file_path, fn, batch_size=128): 
    """Process in batches with real parallelism and live progress tracking"""
    chunk_size = 1024 * 1024 
    merged_set = set()
    file_len = os.path.getsize(file_path)
    
    # Estimate total chunks for progress bar
    estimated_chunks = file_len // chunk_size + 1
    
    with open(file_path, "r", encoding="utf-8") as f:
        with tqdm(total=estimated_chunks, desc=f"Processing {file_path.name}", unit="chunks") as pbar:
            while True:
                # Read batch of chunks
                batch = []
                for _ in range(batch_size):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    batch.append(chunk)
                
                if not batch:
                    break
                
                # Process batch in parallel
                with ProcessPoolExecutor(max_workers=16) as executor:
                    results = list(executor.map(fn, batch))
                
                # Merge results and update progress
                for result in results:
                    merged_set.update(result)
                    pbar.update(1)
    
    return merged_set


def read_file_simple(file_path):
    """Simple file reading"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

if __name__ == "__main__":
    data_dir = Path("./data")
    files = list(data_dir.glob("*.txt"))

    fnames = [f.name for f in files] 
    fnames = [f.stem for f in files]  
    
    print(f"Files: {fnames}")
    

    # Pre-calculate n-gram sets for all files (simple version)
    file_name_ngram_sets = {}
    for fname, f_path in tqdm(zip(fnames, files)):
        print(f"Current file: {fname}")
        file_name_ngram_sets[fname] = batch_and_multithread_process_text(f_path, calculate_ngrams)

    
    # Initialize containment matrix
    containment_matrix = pd.DataFrame(
        index=fnames,
        columns=fnames,
        dtype=float
    )
    
    print(f"\nCalculating pairwise containment...")
    
    def calculate_containment(A, B):
        """Faster containment calculation"""
        # Determine smaller set
        if len(A) <= len(B):
            smaller, larger = A, B
        else:
            smaller, larger = B, A
        
        # Count intersections by iterating smaller set
        intersection_count = sum(1 for item in smaller if item in larger)
        return intersection_count / len(smaller)
    

    total_pairs = len(fnames) * (len(fnames) - 1) // 2  # Only upper triangle
    with tqdm(total=total_pairs, desc="Computing containment matrix") as pbar:
        for i, fname_i in enumerate(fnames):
            for j, fname_j in enumerate(fnames):
                if i == j:
                    containment_matrix.loc[fname_i, fname_j] = 1.0
                elif i < j:  # Only calculate upper triangle
                    set_i = file_name_ngram_sets[fname_i]
                    set_j = file_name_ngram_sets[fname_j]
                    containment = calculate_containment(set_i, set_j)
                    containment_matrix.loc[fname_i, fname_j] = containment
                    containment_matrix.loc[fname_j, fname_i] = containment  # Symmetric
                    pbar.update(1)

    # Replace your display section with this:

    # Save results
    containment_matrix.to_csv("containment_matrix.csv")
    print(f"\nMatrix saved to: containment_matrix.csv")

    # Create file sizes mapping
    file_sizes = {fname: len(file_name_ngram_sets[fname]) for fname in fnames}
    fname_to_path = {f.stem: f for f in files}

    # Sort files by n-gram count (smallest to largest)
    sorted_info = [(fname, file_sizes[fname]) for fname in fnames]
    sorted_info.sort(key=lambda x: x[1])  # Sort by n-gram count
    sorted_fnames = [item[0] for item in sorted_info]

    print("\n" + "="*80)
    print("CONTAINMENT MATRIX (SORTED BY SIZE - UPPER TRIANGLE ONLY)")
    print("Files ordered from smallest to largest")
    print("Shows: How much of ROW file is contained in COLUMN file")
    print("="*80)

    # Display file sizes
    print("\nFile sizes:")
    for fname in sorted_fnames:
        ngram_count = file_sizes[fname]
        size_mb = os.path.getsize(fname_to_path[fname]) / (1024*1024)
        print(f"  {fname:20} {ngram_count:>10,} n-grams ({size_mb:>6.1f} MB)")

    print(f"\nContainment Matrix (Upper Triangle):")
    print("=" * 60)

    # Create header row
    header = "File".ljust(15)
    for j, fname_j in enumerate(sorted_fnames):
        header += f"{fname_j[:8]:>10}"
    print(header)
    print("-" * len(header))

    # Display upper triangle only
    for i, fname_i in enumerate(sorted_fnames):
        row = f"{fname_i[:14]:15}"  # Row label
        
        # Add spaces for lower triangle (empty)
        for j in range(i):
            row += " " * 10
        
        # Add values for upper triangle (i <= j)
        for j in range(i, len(sorted_fnames)):
            fname_j = sorted_fnames[j]
            value = containment_matrix.loc[fname_i, fname_j]
            if i == j:
                row += f"{'1.000':>10}"  # Diagonal
            else:
                row += f"{value*100:>9.1f}%"
        
        print(row)

    print("\nInterpretation:")
    print("- Values show what % of the ROW file is found in the COLUMN file")
    print("- High values (>0.8): ROW file mostly contained in COLUMN file")
    print("- Low values (<0.2): Little overlap between files")
    print("- Only upper triangle shown (smaller → larger containment)")

    
    
    