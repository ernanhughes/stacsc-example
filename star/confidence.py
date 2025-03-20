import logging
import torch
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ConfidenceEvaluator:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.stop_words = set(stopwords.words('english'))  # Load stopwords for analysis

    def calculate_confidence(self, path):
        method = self.config.CONFIDENCE_METHOD
        if method == "model":
            return self.model_based_confidence(path)
        elif method == "heuristic":
            return self.heuristic_confidence(path)
        elif method == "external":
            return self.external_model_confidence(path)
        else:
            logging.warning(f"Unknown confidence method: {method}, defaulting to heuristic.")
            return self.heuristic_confidence(path)

    def model_based_confidence(self, path):
        if len(path) < 500:
            inputs = self.tokenizer(path, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs[:, -1, :].max(dim=-1).values.mean().item()  # Max probability from last token
        else:
            return self.log_probability_confidence(path)

    def log_probability_confidence(self, path):
        """Calculates confidence using log-probabilities, handling length scaling and short text bias."""

        inputs = self.tokenizer(path, return_tensors="pt", padding=True, truncation=True)
        token_ids = inputs["input_ids"]
        seq_length = token_ids.shape[1]  # Number of tokens in sequence

        # Prevent very short sequences from distorting confidence
        min_length = 5  # Set a minimum threshold for meaningful confidence
        if seq_length < min_length:
            logging.warning(f"Text is too short ({seq_length} tokens), confidence may be unreliable.")
            return 0.5  # Return neutral confidence for short texts

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # Get log probabilities

        # Gather the log probabilities of the actual token sequence
        selected_log_probs = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

        # Compute length-adjusted confidence
        avg_log_prob = selected_log_probs.mean()
        length_penalty = seq_length ** 0.5  # Square-root scaling factor to normalize length impact
        normalized_confidence = (avg_log_prob / length_penalty).exp().item()

        # Ensure the confidence stays in a reasonable range
        return max(0.01, min(1.0, normalized_confidence))  # Avoid zero probability issues

    def heuristic_confidence(self, path):
        """Calculates confidence based on heuristic rules."""
        words = path.split()
        num_words = len(words)
        num_sentences = len(re.split(r'[.!?]+', path))  # Count sentences
        num_unique_words = len(set(words))
        stopword_count = sum(1 for word in words if word.lower() in self.stop_words)
        punctuation_count = sum(1 for char in path if char in ".,!?;:")
        word_frequencies = Counter(words)

        # Check for high repetition (word appearing too many times)
        most_common_word_count = word_frequencies.most_common(1)[0][1] if word_frequencies else 0
        repetition_ratio = most_common_word_count / num_words if num_words else 0

        # Calculate scores
        sentence_length_score = min(1.0, num_words / (num_sentences + 1))  # Penalize long sentences
        unique_word_ratio = num_unique_words / num_words if num_words else 0  # Higher is better
        punctuation_score = min(1.0, punctuation_count / (num_words + 1))  # More punctuation could indicate lower confidence
        stopword_ratio = stopword_count / num_words if num_words else 0  # Moderate stopword use is ideal
        repetition_penalty = max(0, 1 - (repetition_ratio * 5))  # High repetition reduces confidence

        # Combine scores (weighted sum, fine-tune as needed)
        confidence_score = (
            (0.3 * unique_word_ratio) +
            (0.2 * repetition_penalty) +
            (0.2 * sentence_length_score) +
            (0.2 * (1 - punctuation_score)) +  # More punctuation = lower confidence
            (0.1 * (1 - abs(stopword_ratio - 0.4)))  # Ideal stopword ratio ~40%
        )

        return max(0.0, min(1.0, confidence_score))  # Keep score in [0,1] range

    def external_model_confidence(self, path):
        """Placeholder for external confidence verification."""
        return 0.5  # Dummy value
