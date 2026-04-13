"""Main CLI module for N-gram Predictor application."""
from normalizer import Normalizer
from ngram_model import NGramModel
from predictor import Predictor
from evaluator import Evaluator
from predictor_ui import PredictorUI
from data_prep import get_training_texts, save_normalized_books


test_texts = [
    "To Sherlock Holmes she is always _the_ woman. I have seldom heard him mention her under any other name."
]


def main():
    """
    Main CLI loop.
    
    Initializes and coordinates:
    - Normalizer: used by Data Prep, Inference, and Evaluator
    - NGramModel: core model with vocab, counts, probabilities, and lookup
    - Predictor: uses NGramModel and Normalizer for predictions
    - Evaluator: optional extra credit for model evaluation
    - PredictorUI: optional extra credit for interactive interface
    """
    print("=" * 50)
    print("N-Gram Predictor")
    print("=" * 50)
    
    # Initialize core components
    normalizer = Normalizer()
    model = NGramModel(n=2)  # Default bigram model
    predictor = Predictor(model, normalizer)
    
    # Optional: Initialize extra credit components
    evaluator = Evaluator(model, normalizer)
    ui = PredictorUI(predictor)
    
    # CLI loop
    while True:
        print("\nOptions:")
        print("1. Train model")
        print("2. Make prediction")
        print("3. Evaluate model (extra credit)")
        print("4. Interactive UI (extra credit)")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            print("Training model...")
            normalized_texts = get_training_texts()
            model.build_vocab(normalized_texts)
            model.build_counts_and_probabilities(normalized_texts)
            save_normalized_books()
            model.save_model("model.json")
            print("Model trained and saved.")
            print("Normalized training files saved to data/normalized.")
        elif choice == "2":
            context = input("Enter context: ").strip()
            if context:
                predictions = predictor.predict_next(context, 5)
                print("Predictions:", predictions)
            else:
                print("No context provided.")
        elif choice == "3":
            print("Evaluating model...")
            metrics = evaluator.run(test_texts)
            print("Perplexity:", metrics['perplexity'])
        elif choice == "4":
            print("Launching UI...")
            ui.run()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
