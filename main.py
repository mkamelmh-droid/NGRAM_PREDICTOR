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
from config import TOP_K

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "train_raw")
TRAIN_TOKENS_FILE = os.path.join(DATA_DIR, "train_tokens.txt")
MODEL_PATH = os.path.join(DATA_DIR, "model.json")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")
ENV_PATH = os.path.join(ROOT_DIR, "config", ".env")


def load_environment() -> None:
    """Load environment variables from config/.env before any pipeline step."""
    load_dotenv(dotenv_path=ENV_PATH)


def run_dataprep() -> None:
    """Run Normalizer to produce train_tokens.txt from raw text files."""
    print("=" * 50)
    print("Step: dataprep")
    print("=" * 50)

    if os.path.isdir(RAW_DATA_DIR) and any(f.endswith('.txt') for f in os.listdir(RAW_DATA_DIR)):
        input_folder = RAW_DATA_DIR
    else:
        input_folder = DATA_DIR

    normalizer = Normalizer()
    normalizer.process_and_save(input_folder=input_folder, output_file=TRAIN_TOKENS_FILE)

    print(f"Produced training tokens: {TRAIN_TOKENS_FILE}")


def run_model() -> None:
    """Run NGramModel to build and save model.json and vocab.json."""
    print("=" * 50)
    print("Step: model")
    print("=" * 50)

    if not os.path.isfile(TRAIN_TOKENS_FILE):
        raise FileNotFoundError(
            f"Training tokens not found: {TRAIN_TOKENS_FILE}. "
            "Run `python main.py --step dataprep` first."
        )

    model = NGramModel()
    model.build_vocab(TRAIN_TOKENS_FILE)
    model.build_counts_and_probabilities(TRAIN_TOKENS_FILE)
    model.save_model(MODEL_PATH)
    model.save_vocab(VOCAB_PATH)

    print(f"Model saved: {MODEL_PATH}")
    print(f"Vocab saved: {VOCAB_PATH}")


def run_inference(k: int = None) -> None:
    """Start the interactive CLI loop for top-k prediction."""
    print("=" * 50)
    print("Step: inference")
    print("=" * 50)

    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VOCAB_PATH):
        raise FileNotFoundError(
            f"Model or vocab file missing. Run `python main.py --step model` first."
        )

    predictor = Predictor(MODEL_PATH, VOCAB_PATH)
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
