import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fine_tuner import FineTuner

class ModelManager:
    def __init__(self, config, db, fine_tune_method='gradient'):
        self.config = config
        self.db = db
        self.fine_tuner = FineTuner(config, db, fine_tune_method)
        self.load_model()

    def fine_tune_model(self, training_text):
        """Fine-tunes the model using the selected method."""
        self.fine_tuner.fine_tune_model(training_text)

    def save_fine_tuned_model(self):
        """Saves the fine-tuned model."""
        self.fine_tuner.save_fine_tuned_model()

    def load_model(self):
        """Loads the latest fine-tuned model."""
        self.fine_tuner._load_model()
        self.model = self.fine_tuner.model
        self.tokenizer = self.fine_tuner.tokenizer

