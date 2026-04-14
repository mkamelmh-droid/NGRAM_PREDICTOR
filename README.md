# N-gram Predictor

An N-gram language model for next-word prediction trained on Sherlock Holmes novels.

## Overview

This project implements a backoff-based N-gram predictor that can suggest the most likely next words given a phrase. It includes both a command-line interface (CLI) and an optional Streamlit web UI.

## Quick Start

Get predictions in seconds!

## Features

- **Data Preparation**: Processes raw text files into tokenized training data.
- **Model Training**: Builds N-gram probability tables with backoff.
- **Inference**: Predicts top-k next words for any input phrase.
- **CLI**: Command-line interface with `--step` options.
- **Web UI**: Browser-based interface using Streamlit (extra credit).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mkamelmh-droid/NGRAM_PREDICTOR.git
   cd NGRAM_PREDICTOR
   ```

2. Create and activate an Anaconda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ngram-env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Populate `config/.env` with your local paths and settings.

5. Download the four training books and place them in `data/raw/train/`:
   - study_in_scarlet.txt
   - sign_of_the_four.txt
   - hound_of_the_baskervilles.txt
   - valley_of_fear.txt

## Usage

### CLI

Run the full pipeline:
```bash
python main.py --step all
```

Or run individual steps:
- `python main.py --step dataprep` - Prepare training data
- `python main.py --step model` - Train the model
- `python main.py --step inference` - Interactive prediction

### Web UI (Extra Credit)

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Project Structure

```
NGRAM_PREDICTOR/
├── main.py                 # CLI entry point
├── app.py                  # Streamlit web UI
├── config.py               # Configuration constants
├── normalizer.py           # Text normalization
├── ngram_model.py          # N-gram model implementation
├── predictor.py            # Prediction interface
├── data_prep.py            # Data preparation utilities
├── data/
│   ├── raw/
│   │   ├── train/          # Raw training texts
│   │   └── eval/           # Raw evaluation texts (extra credit)
│   ├── processed/
│   │   ├── train_tokens.txt # Tokenized training data
│   │   └── eval_tokens.txt  # Tokenized evaluation data (extra credit)
│   └── model/
│       ├── model.json      # Trained model
│       └── vocab.json      # Vocabulary
├── config/
│   └── .env                # Environment variables
├── requirements.txt        # Python dependencies
├── environment.yml         # Anaconda environment
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Configuration

All settings are loaded from `config/.env`. Key variables:
- `NGRAM_ORDER`: N-gram order (default: 4)
- `UNK_THRESHOLD`: Frequency threshold for unknown words (default: 1)
- `TOP_K`: Default number of predictions (default: 5)

## Configuration

All settings are loaded from `config/.env`. Required variables:
- `TRAIN_RAW_DIR=data/raw/train/`
- `EVAL_RAW_DIR=data/raw/eval/`
- `TRAIN_TOKENS=data/processed/train_tokens.txt`
- `EVAL_TOKENS=data/processed/eval_tokens.txt`
- `MODEL=data/model/model.json`
- `VOCAB=data/model/vocab.json`
- `UNK_THRESHOLD=1`
- `TOP_K=5`
- `NGRAM_ORDER=4`

## License

This project is for educational purposes.