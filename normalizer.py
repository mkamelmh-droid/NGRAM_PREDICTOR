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
        Handles multiple books concatenated together.
        
        Args:
            text: Raw text from Project Gutenberg (may contain multiple books)
            
        Returns:
            Text with all headers and footers removed
        """
        # Find all START markers
        start_pattern = r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
        start_matches = list(re.finditer(start_pattern, text, re.IGNORECASE | re.DOTALL))
        
        # Find all END markers  
        end_pattern = r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
        end_matches = list(re.finditer(end_pattern, text, re.IGNORECASE | re.DOTALL))
        
        if start_matches and end_matches:
            # Remove headers and footers by keeping only content between markers
            result_parts = []
            for start_match, end_match in zip(start_matches, end_matches):
                # Get content after each START marker and before each END marker
                content = text[start_match.end():end_match.start()].strip()
                if content:
                    result_parts.append(content)
            
            if result_parts:
                return '\n\n'.join(result_parts)
        
        # Fallback: try simpler markers
        start_pattern = r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK"
        end_pattern = r"END OF (THIS|THE) PROJECT GUTENBERG EBOOK"
        
        start_matches = list(re.finditer(start_pattern, text, re.IGNORECASE))
        end_matches = list(re.finditer(end_pattern, text, re.IGNORECASE))
        
        if start_matches and end_matches:
            result_parts = []
            for start_match, end_match in zip(start_matches, end_matches):
                content = text[start_match.end():end_match.start()].strip()
                if content:
                    result_parts.append(content)
            
            if result_parts:
                return '\n\n'.join(result_parts)
        
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
        
        # Step 4: Sentence tokenize BEFORE normalization (before punctuation is removed)
        sentences = self.sentence_tokenize(stripped_text)
        
        # Step 3 & 5: Normalize and word tokenize
        tokenized_sentences = []
        for sentence in sentences:
            # Normalize the sentence
            normalized_sentence = self.normalize(sentence)
            # Word tokenize the normalized sentence
            tokens = self.word_tokenize(normalized_sentence)
            if tokens:  # Skip empty sentences
                tokenized_sentences.append(' '.join(tokens))
        
        # Step 6: Write output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for tokenized_sentence in tokenized_sentences:
                f.write(tokenized_sentence + '\n')
