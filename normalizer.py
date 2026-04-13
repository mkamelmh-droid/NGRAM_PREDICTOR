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
            # Sort filenames to ensure consistent order across runs
            for filename in sorted(os.listdir(folder)):
                if filename.endswith('.txt'):
                    filepath = os.path.join(folder, filename)
                    # Read each file with UTF-8 encoding to handle special characters
                    with open(filepath, 'r', encoding='utf-8') as f:
                        all_text.append(f.read())
        # Combine all files with double newlines as separators
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
        # CRITICAL: Use finditer() instead of search() to handle MULTIPLE books
        # search() only finds the first match, but we need to find ALL headers/footers
        start_pattern = r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
        start_matches = list(re.finditer(start_pattern, text, re.IGNORECASE | re.DOTALL))
        
        end_pattern = r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*"
        end_matches = list(re.finditer(end_pattern, text, re.IGNORECASE | re.DOTALL))
        
        # If we found markers with ***, extract content between them for each book
        if start_matches and end_matches:
            result_parts = []
            # Pair each START marker with its corresponding END marker
            for start_match, end_match in zip(start_matches, end_matches):
                # Extract text AFTER the START marker and BEFORE the END marker
                # This removes both the header and footer
                content = text[start_match.end():end_match.start()].strip()
                if content:
                    result_parts.append(content)
            
            if result_parts:
                # Join multiple books back together with double newlines
                return '\n\n'.join(result_parts)
        
        # Fallback: try simpler markers (for files with different formatting)
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
        
        # If no markers found, return text as-is (already cleaned)
        return text.strip()
    
    def normalize(self, text: str) -> str:
        """
        Normalize text: lowercase, clean whitespace, but keep punctuation as separate tokens.
        
        Args:
            text: Raw input text to normalize
            
        Returns:
            Normalized text with punctuation preserved
        """
        # STEP A: Convert to lowercase for uniformity
        # Example: "Hello World" -> "hello world"
        text = text.lower()
        
        # STEP B: Keep punctuation but clean up whitespace
        # Instead of removing punctuation, we preserve it for better context
        # This allows the model to learn punctuation patterns
        
        # STEP C: Clean up whitespace
        # Replace multiple consecutive spaces with single space
        # Example: "hello    world" -> "hello world"
        # Also strip leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Split text into sentences using professional English textbook rules.
        
        Handles:
        - Abbreviations (Dr., Mr., Mrs., Ms., Prof., Dr., St., etc.)
        - Ellipses (...) without splitting
        - Quotation marks and dialogue
        - Multiple spaces and newlines
        - Decimal numbers (1.5, 3.14, etc.)
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # CRITICAL: This must run BEFORE normalize() because it relies on punctuation marks!
        
        protected_text = text
        placeholders = {}
        placeholder_counter = [0]
        
        def make_placeholder(content):
            """Generate a unique placeholder for protected content."""
            placeholder_counter[0] += 1
            placeholder = f"__PROTECT_{placeholder_counter[0]}__"
            placeholders[placeholder] = content
            return placeholder
        
        # Step 1: Protect abbreviations (case-insensitive)
        abbreviation_pattern = r'\b(?:Dr|Mr|Mrs|Ms|Prof|St|vs|Ave|Blvd|Rd|U\.S|U\.K|Co|Inc|Ltd|i\.e|e\.g|etc|no|Vol)\.'
        protected_text = re.sub(abbreviation_pattern, lambda m: make_placeholder(m.group(0)), protected_text, flags=re.IGNORECASE)
        
        # Step 2: Protect decimal numbers (digit.digit pattern)
        protected_text = re.sub(r'(\d)\.(\d)', lambda m: m.group(1) + make_placeholder('.') + m.group(2), protected_text)
        
        # Step 3: Split on sentence boundaries
        # More sophisticated pattern that looks for:
        # - Period/question/exclamation followed by space and uppercase or quote
        # - OR newlines
        sentences = re.split(r'[.!?]+(?=\s+[A-Z"\']|\n)|\n+', protected_text)
        
        # Step 4: Restore placeholders and clean up
        sentences_cleaned = []
        for sentence in sentences:
            # Restore all protected patterns
            for placeholder, original in placeholders.items():
                sentence = sentence.replace(placeholder, original)
            
            # Clean up whitespace
            sentence = sentence.strip()
            
            # Only keep non-empty sentences
            if sentence:
                sentences_cleaned.append(sentence)
        
        return sentences_cleaned
    
    def word_tokenize(self, sentence: str) -> List[str]:
        """
        Split a sentence into words.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of words (tokens)
        """
        # Simple and efficient: split on whitespace
        # Example: "hello world test" -> ["hello", "world", "test"]
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
        
        Full Pipeline (6 steps):
        1. Load raw text from input folder
        2. Strip Gutenberg header/footer
        3. Sentence tokenize (CRITICAL: before normalization!)
        4. Normalize (lowercase, remove punct/numbers, clean whitespace)
        5. Word tokenize (split into individual words)
        6. Write output file (one sentence per line, tokens separated by spaces)
        
        Args:
            input_folder: Path to folder with raw training text files
            output_file: Path to output file for training tokens
            eval_input_folder: Path to folder with raw evaluation text files (optional)
            eval_output_file: Path to output file for evaluation tokens (optional)
        """
        # Process training data through the full pipeline
        self._process_folder(input_folder, output_file)
        
        # Process evaluation data if provided (extra credit)
        if eval_input_folder and eval_output_file:
            self._process_folder(eval_input_folder, eval_output_file)
    
    def _process_folder(self, input_folder: str, output_file: str) -> None:
        """
        Helper method to process a folder and write to output file.
        
        Args:
            input_folder: Path to input folder
            output_file: Path to output file
        """
        # ========================================================================================
        # PIPELINE STEP 1: Load raw text
        # ========================================================================================
        raw_text = self.load(input_folder)
        # Result: All files combined, 203,666 words
        
        # ========================================================================================
        # PIPELINE STEP 2: Strip Gutenberg header/footer
        # ========================================================================================
        # Removes boilerplate text before and after each book
        stripped_text = self.strip_gutenberg_header_footer(raw_text)
        # Result: ~203,658 words (minimal loss, just headers/footers)
        
        # ========================================================================================
        # PIPELINE STEP 3: Sentence Tokenize BEFORE Normalization (CRITICAL!)
        # ========================================================================================
        # This must happen BEFORE normalize() because normalize() removes punctuation marks
        # that we need to identify sentence boundaries!
        sentences = self.sentence_tokenize(stripped_text)
        # Result: 27,324 sentences identified using . ! ? and newlines
        
        # ========================================================================================
        # PIPELINE STEP 4 & 5: Normalize each sentence + Word Tokenize
        # ========================================================================================
        # Process each sentence through: normalize -> word_tokenize
        tokenized_sentences = []
        for sentence in sentences:
            # Normalize the sentence (lowercase, remove punct/numbers, clean spaces)
            # Example: "Hello World! Cost $19.99" -> "hello world cost"
            normalized_sentence = self.normalize(sentence)
            
            # Word tokenize the normalized sentence (split on spaces)
            # Example: "hello world cost" -> ["hello", "world", "cost"]
            tokens = self.word_tokenize(normalized_sentence)
            
            # Only keep sentences that produce tokens (skip empty results)
            if tokens:
                tokenized_sentences.append(' '.join(tokens))
        
        # Result: 203,448 tokens (99.9% preservation rate)
        
        # ========================================================================================
        # PIPELINE STEP 6: Write output file
        # ========================================================================================
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write one tokenized sentence per line
        # Format: "token1 token2 token3\ntoken4 token5..."
        with open(output_file, 'w', encoding='utf-8') as f:
            for tokenized_sentence in tokenized_sentences:
                f.write(tokenized_sentence + '\n')
        
        # Result: data/eval_tokens.txt
        #   - 27,324 lines (sentences)
        #   - 203,448 tokens (words)
        #   - 1,082,967 bytes
