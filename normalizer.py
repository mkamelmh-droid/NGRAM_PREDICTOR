"""Text normalization module for N-gram predictor."""
import os
import re
from typing import List


class Normalizer:
    """Normalizes text for preprocessing and inference."""
    
    def __init__(self):
        """Initialize the Normalizer."""
        pass
    
    def load(self, folder: str) -> str:
        """
        Load all .txt files from a folder.
        
        Args:
            folder: Path to folder containing .txt files
            
        Returns:
            Concatenated text from all files
        """
        all_text = []
        if os.path.isdir(folder):
            for filename in sorted(os.listdir(folder)):
                if filename.endswith('.txt'):
                    filepath = os.path.join(folder, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        all_text.append(f.read())
        return '\n\n'.join(all_text)
    
    def strip_gutenberg_header_footer(self, text: str) -> str:
        """
        Remove Project Gutenberg header and footer.
        
        Args:
            text: Raw text from Project Gutenberg
            
        Returns:
            Text with header and footer removed
        """
        # Try to find the standard *** markers first
        start_match = re.search(
            r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
            text,
            re.IGNORECASE | re.DOTALL
        )
        end_match = re.search(
            r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if start_match and end_match:
            return text[start_match.end():end_match.start()].strip()
        
        # Fallback to simpler markers
        start_match = re.search(
            r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK",
            text,
            re.IGNORECASE
        )
        end_match = re.search(
            r"END OF (THIS|THE) PROJECT GUTENBERG EBOOK",
            text,
            re.IGNORECASE
        )
        
        if start_match and end_match:
            return text[start_match.end():end_match.start()].strip()
        
        return text.strip()
    
    def normalize(self, text: str) -> str:
        """
        Normalize text: lowercase, remove punctuation, remove numbers, remove extra whitespace.
        
        Args:
            text: Raw input text to normalize
            
        Returns:
            Normalized text
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation and numbers, keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence tokenizer: split on . ! ? followed by space and capital letter
        # Also split on newlines
        sentences = re.split(r'[.!?]+\s+|[\n]+', text)
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def word_tokenize(self, sentence: str) -> List[str]:
        """
        Split a sentence into words.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of words (tokens)
        """
        # Split on whitespace
        tokens = sentence.split()
        return tokens
    
    def process_and_save(
        self,
        input_folder: str,
        output_file: str,
        eval_input_folder: str = None,
        eval_output_file: str = None
    ) -> None:
        """
        Process raw text through the full pipeline and save to output files.
        
        Pipeline:
        1. Load raw text from input folder
        2. Strip Gutenberg header/footer
        3. Normalize (lowercase, remove punctuation, remove numbers, remove extra whitespace)
        4. Sentence tokenize
        5. Word tokenize
        6. Write output file (one sentence per line, tokens separated by spaces)
        
        Args:
            input_folder: Path to folder with raw training text files
            output_file: Path to output file for training tokens
            eval_input_folder: Path to folder with raw evaluation text files (optional)
            eval_output_file: Path to output file for evaluation tokens (optional)
        """
        # Process training data
        self._process_folder(input_folder, output_file)
        
        # Process evaluation data if provided
        if eval_input_folder and eval_output_file:
            self._process_folder(eval_input_folder, eval_output_file)
    
    def _process_folder(self, input_folder: str, output_file: str) -> None:
        """
        Helper method to process a folder and write to output file.
        
        Args:
            input_folder: Path to input folder
            output_file: Path to output file
        """
        # Step 1: Load raw text
        raw_text = self.load(input_folder)
        
        # Step 2: Strip Gutenberg header/footer
        stripped_text = self.strip_gutenberg_header_footer(raw_text)
        
        # Step 3: Normalize
        normalized_text = self.normalize(stripped_text)
        
        # Step 4 & 5: Sentence tokenize and word tokenize
        sentences = self.sentence_tokenize(normalized_text)
        tokenized_sentences = [
            ' '.join(self.word_tokenize(sentence))
            for sentence in sentences
            if self.word_tokenize(sentence)  # Skip empty sentences
        ]
        
        # Step 6: Write output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for tokenized_sentence in tokenized_sentences:
                f.write(tokenized_sentence + '\n')
