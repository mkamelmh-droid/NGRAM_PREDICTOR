"""Data preparation module for N-gram predictor.

This module loads the full Project Gutenberg texts for the four training novels and
normalizes them using the Normalizer.
"""
import os
import re
import urllib.request
from typing import Dict, List
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


def ensure_data_dir(data_dir: str = "data") -> None:
    os.makedirs(data_dir, exist_ok=True)


def download_text(url: str, filepath: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def strip_gutenberg_header_footer(text: str) -> str:
    start_match = re.search(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    end_match = re.search(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE | re.DOTALL)
    if start_match and end_match:
        return text[start_match.end():end_match.start()].strip()

    start_match = re.search(r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK", text, re.IGNORECASE)
    end_match = re.search(r"END OF (THIS|THE) PROJECT GUTENBERG EBOOK", text, re.IGNORECASE)
    if start_match and end_match:
        return text[start_match.end():end_match.start()].strip()

    return text.strip()


def load_book(book_key: str, data_dir: str = "data") -> str:
    source = BOOK_SOURCES[book_key]
    filepath = os.path.join(data_dir, source["filename"])
    ensure_data_dir(data_dir)
    if not os.path.exists(filepath):
        download_text(source["url"], filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return strip_gutenberg_header_footer(raw_text)


def load_training_sources(data_dir: str = "data") -> Dict[str, str]:
    return {key: load_book(key, data_dir) for key in BOOK_SOURCES}


def normalize_sources(sources: Dict[str, str]) -> Dict[str, str]:
    """Normalize multiple raw source strings using the Normalizer."""
    normalizer = Normalizer()
    return {key: normalizer.normalize(text) for key, text in sources.items()}


def save_normalized_books(data_dir: str = "data", normalized_dir: str = "data/normalized") -> Dict[str, str]:
    """Save normalized text files for each Gutenberg source."""
    raw_sources = load_training_sources(data_dir)
    normalized_sources = normalize_sources(raw_sources)
    ensure_data_dir(normalized_dir)
    for key, normalized_text in normalized_sources.items():
        out_path = os.path.join(normalized_dir, f"{key}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(normalized_text)
    return normalized_sources


def get_training_texts(data_dir: str = "data", normalized_dir: str = "data/normalized") -> List[str]:
    """Return the normalized training corpus from the Gutenberg source texts."""
    normalized_sources = save_normalized_books(data_dir, normalized_dir)
    return [normalized_sources[key] for key in BOOK_SOURCES]
