"""Main CLI module for N-gram Predictor application."""
from normalizer import Normalizer
from ngram_model import NGramModel
from predictor import Predictor
from evaluator import Evaluator
from predictor_ui import PredictorUI


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
            # TODO: Implement training
            pass
        elif choice == "2":
            print("Making prediction...")
            # TODO: Implement prediction
            pass
        elif choice == "3":
            print("Evaluating model...")
            # TODO: Implement evaluation
            pass
        elif choice == "4":
            print("Starting interactive UI...")
            # TODO: Implement UI
            pass
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
