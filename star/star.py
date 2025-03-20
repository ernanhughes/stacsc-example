import os
import logging

from config import Config
from database import Database
from model_manager import ModelManager
from confidence import ConfidenceEvaluator



class STaR:
    def __init__(self):
        self.config = Config()
        self.db = Database()
        self.model_save_path = os.path.join(self.config.MODEL_SAVE_DIR, "latest")
        self.model_manager = ModelManager(self.config, self.db)
        self.confidence_evaluator = ConfidenceEvaluator(self.config, self.model_manager.model,
                                                        self.model_manager.tokenizer)

    def evaluate_paths(self, question, paths):
        best_path = None
        best_confidence = 0.0
        confidence_list = []  # Store confidence values to compute threshold dynamically

        for path in paths:
            confidence = self.confidence_evaluator.calculate_confidence(path)
            confidence_list.append(confidence)

            is_correct = confidence > self.config.CONFIDENCE_THRESHOLD  # Use configurable threshold
            self.db.save_path(question, path, is_correct, confidence)

            if confidence > best_confidence:
                best_confidence = confidence
                best_path = path

        if best_path:
            self.db.update_best_path(question, best_path)
            self.model_manager.save_fine_tuned_model()  # Save model after determining best path

        logging.info(f"Best answer selected with confidence {best_confidence}")
        return best_path
