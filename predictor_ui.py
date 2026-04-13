"""Predictor UI module for interactive user interface (extra credit)."""
from predictor import Predictor


class PredictorUI:
    """Interactive user interface for the N-gram predictor."""
    
    def __init__(self, predictor: Predictor):
        """
        Initialize the PredictorUI.
        
        Args:
            predictor: Predictor instance to use for predictions
        """
        self.predictor = predictor
    
    def run(self) -> None:
        """
        Run the interactive UI loop.
        """
        try:
            import streamlit as st
        except ImportError:
            print("Streamlit not installed. Please install streamlit to use the UI.")
            return
        
        st.title("N-Gram Next-Word Predictor")
        
        context = st.text_input("Enter context text:", "")
        k = st.slider("Number of predictions:", 1, 10, 5)
        
        if st.button("Predict"):
            if context:
                predictions = self.predictor.predict_next(context, k)
                st.write("Top predictions:")
                for i, word in enumerate(predictions, 1):
                    st.write(f"{i}. {word}")
            else:
                st.write("Please enter some context.")
