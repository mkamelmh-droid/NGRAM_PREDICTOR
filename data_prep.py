"""Data preparation module for N-gram predictor.

This module downloads the four Project Gutenberg training novels and processes
them through the Normalizer pipeline to create a single tokenized training file.
"""
import os
import urllib.request
from typing import Dict
from normalizer import Normalizer

BOOK_SOURCES = {
    "study_in_scarlet": {
        "url": "https://www.gutenberg.org/files/244/244-0.txt",
        "filename": "study_in_scarlet.txt",
    },
    "sign_of_the_four": {
        "url": "https://www.gutenberg.org/files/2097/2097-0.txt",
        "filename": "sign_of_the_four.txt",
    },
    "hound_of_the_baskervilles": {
        "url": "https://www.gutenberg.org/files/2852/2852-0.txt",
        "filename": "hound_of_the_baskervilles.txt",
    },
    "valley_of_fear": {
        "url": "https://www.gutenberg.org/files/3289/3289-0.txt",
        "filename": "valley_of_fear.txt",
    },
}

# Default folders
RAW_DATA_DIR = "data/train_raw"
TOKENS_OUTPUT_FILE = "data/train_tokens.txt"
EVAL_RAW_DATA_DIR = "data/eval_raw"
EVAL_TOKENS_OUTPUT_FILE = "data/eval_tokens.txt"


def ensure_data_dir(data_dir: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(data_dir, exist_ok=True)


def download_text(url: str, filepath: str) -> str:
    """Download text from URL and save to filepath."""
    print(f"Downloading {filepath}...")
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved to {filepath}")
    return text


def download_training_books(data_dir: str = RAW_DATA_DIR) -> None:
    """Download all training books to the specified directory."""
    ensure_data_dir(data_dir)
    for book_key, source in BOOK_SOURCES.items():
        filepath = os.path.join(data_dir, source["filename"])
        if not os.path.exists(filepath):
            download_text(source["url"], filepath)
        else:
            print(f"File already exists: {filepath}")


def process_training_data(
    raw_dir: str = RAW_DATA_DIR,
    output_file: str = TOKENS_OUTPUT_FILE,
    eval_raw_dir: str = None,
    eval_output_file: str = None
) -> None:
    """
    Download training books and process them through the normalization pipeline.
    
    Creates a single output file with all training books combined:
    - One sentence per line
    - Tokens separated by spaces
    - Fully normalized (lowercase, no punctuation, no numbers)
    
    Args:
        raw_dir: Directory to download raw training books to
        output_file: Output file path for tokenized training data
        eval_raw_dir: Optional directory for evaluation books
        eval_output_file: Optional output file for evaluation tokens
    """
    # Download the books
    print("=" * 50)
    print("Downloading training books...")
    print("=" * 50)
    download_training_books(raw_dir)
    
    # Process through normalization pipeline
    print("\n" + "=" * 50)
    print("Processing through normalization pipeline...")
    print("=" * 50)
    normalizer = Normalizer()
    normalizer.process_and_save(
        input_folder=raw_dir,
        output_file=output_file,
        eval_input_folder=eval_raw_dir,
        eval_output_file=eval_output_file
    )
    print(f"Training tokens written to: {output_file}")
    if eval_output_file:
        print(f"Evaluation tokens written to: {eval_output_file}")


def get_training_texts(output_file: str = TOKENS_OUTPUT_FILE) -> list:
    """
    Load tokenized training texts from output file.
    
    Args:
        output_file: Path to tokenized output file
        
    Returns:
        List of tokenized sentences (one per line from file)
    """
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Training tokens file not found: {output_file}")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    
    return sentences
