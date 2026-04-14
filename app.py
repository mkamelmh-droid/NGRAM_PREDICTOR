"""
Streamlit UI for N-gram Predictor (Extra Credit Module 5)
Provides a browser-based interface for next-word predictions.

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from predictor import Predictor
from config import TOP_K

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.json")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

# Load predictor (cached to avoid reloading on every interaction)
@st.cache_resource
def load_predictor():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(VOCAB_PATH):
        st.error("Model files not found. Please run the CLI with --step model first.")
        st.stop()
    return Predictor(MODEL_PATH, VOCAB_PATH)

def main():
    st.title("N-gram Next-Word Predictor")
    st.markdown("Enter a phrase to get the top-k most likely next words.")

    # Load the predictor
    predictor = load_predictor()

    # Input widgets
    phrase = st.text_input("Enter a phrase:", placeholder="e.g., holmes said")
    k = st.number_input("Number of predictions (k):", min_value=1, max_value=20, value=TOP_K)

    # Predict button
    if st.button("Predict Next Words"):
        if phrase.strip():
            predictions = predictor.predict_next(phrase, k)
            st.success(f"Predictions for '{phrase}':")
            st.write(predictions)
        else:
            st.warning("Please enter a phrase.")

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit. Model trained on Sherlock Holmes novels.")

if __name__ == "__main__":
    main()