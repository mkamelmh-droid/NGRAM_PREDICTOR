"""Main CLI module for N-gram Predictor application."""
import argparse
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path=None):
        """Fallback no-op dotenv loader if python-dotenv is unavailable."""
        return False

from normalizer import Normalizer
from ngram_model import NGramModel
from predictor import Predictor
from config import (
    TRAIN_RAW_DIR, TRAIN_TOKENS, MODEL, VOCAB, TOP_K,
    EVAL_RAW_DIR, EVAL_TOKENS
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "train")
TOKENS_OUTPUT_FILE = os.path.join(DATA_DIR, "processed", "train_tokens.txt")
MODEL_PATH = os.path.join(DATA_DIR, "model", "model.json")
VOCAB_PATH = os.path.join(DATA_DIR, "model", "vocab.json")
ENV_PATH = os.path.join(ROOT_DIR, "config", ".env")


def load_environment() -> None:
    """Load environment variables from config/.env before any pipeline step."""
    load_dotenv(dotenv_path=ENV_PATH)


def run_dataprep() -> None:
    """Run Normalizer to produce train_tokens.txt from raw text files."""
    print("=" * 50)
    print("Step: dataprep")
    print("=" * 50)

    if os.path.isdir(TRAIN_RAW_DIR) and any(f.endswith('.txt') for f in os.listdir(TRAIN_RAW_DIR)):
        input_folder = TRAIN_RAW_DIR
    else:
        input_folder = RAW_DATA_DIR

    normalizer = Normalizer()
    normalizer.process_and_save(input_folder=input_folder, output_file=TRAIN_TOKENS)

    print(f"Produced training tokens: {TRAIN_TOKENS}")


def run_model() -> None:
    """Run NGramModel to build and save model.json and vocab.json."""
    print("=" * 50)
    print("Step: model")
    print("=" * 50)

    if not os.path.isfile(TRAIN_TOKENS):
        raise FileNotFoundError(
            f"Training tokens not found: {TRAIN_TOKENS}. "
            "Run `python main.py --step dataprep` first."
        )

    model = NGramModel()
    model.build_vocab(TRAIN_TOKENS)
    model.build_counts_and_probabilities(TRAIN_TOKENS)
    model.save_model(MODEL)
    model.save_vocab(VOCAB)

    print(f"Model saved: {MODEL}")
    print(f"Vocab saved: {VOCAB}")


def run_inference(k: int = None) -> None:
    """Start the interactive CLI loop for top-k prediction."""
    print("=" * 50)
    print("Step: inference")
    print("=" * 50)

    if not os.path.isfile(MODEL) or not os.path.isfile(VOCAB):
        raise FileNotFoundError(
            f"Model or vocab file missing. Run `python main.py --step model` first."
        )

    predictor = Predictor(MODEL, VOCAB)
    if k is None:
        k = TOP_K

    try:
        while True:
            user_input = input("> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit"}:
                print("Goodbye.")
                break

            predictions = predictor.predict_next(user_input, k)
            print(f"Predictions: {predictions}")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="N-gram predictor command-line interface."
    )
    parser.add_argument(
        "--step",
        choices=["dataprep", "model", "inference", "all"],
        required=True,
        help="Pipeline step to run: dataprep, model, inference, or all."
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()

    if args.step == "dataprep":
        run_dataprep()
    elif args.step == "model":
        run_model()
    elif args.step == "inference":
        run_inference()
    elif args.step == "all":
        run_dataprep()
        run_model()
        run_inference()
    else:
        raise ValueError(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()
