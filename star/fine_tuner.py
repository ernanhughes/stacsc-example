import logging
import os
import torch
import peft
import sqlite3
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig


class FineTuner:
    def __init__(self, config, db, fine_tune_method='gradient'):
        self.config = config
        self.db = db
        self.fine_tune_method = fine_tune_method  # Options: 'gradient', 'lora', 'full_model'

        # ✅ Automatically detect the best available device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self._load_model()

    def _load_model(self):
        """Loads the latest fine-tuned model or falls back to the base model."""
        latest_model = self.db.get_latest_fine_tuned_model()
        gradient_checkpoint = os.path.join(latest_model, "gradient_checkpoint.pt")

        if latest_model and os.path.exists(latest_model):
            try:
                logging.info("Loading fine-tuned model...")
                self.tokenizer = AutoTokenizer.from_pretrained(latest_model)
                self.model = AutoModelForCausalLM.from_pretrained(latest_model).to(self.device)

                if self.fine_tune_method == "gradient" and os.path.exists(gradient_checkpoint):
                    self._load_gradient_checkpoint(gradient_checkpoint)

                logging.info("Fine-tuned model loaded successfully.")
                return
            except Exception as e:
                logging.error(f"Failed to load fine-tuned model. Error: {e}")

        # Load base model as fallback
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.MODEL_NAME).to(self.device)
        logging.info("Loaded default model.")

    def _load_gradient_checkpoint(self, checkpoint_path):
        """Loads optimizer state (gradients) from checkpoint and moves to correct device."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)  # ✅ Load directly to correct device
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)

            # Convert gradients back to FP32 after loading
            optimizer_state = {k: v.to(self.device).float() for k, v in checkpoint["optimizer_state_dict"].items()}
            self.optimizer.load_state_dict(optimizer_state)

            logging.info(f"Loaded optimizer state (FP16 gradients converted back to FP32) from {checkpoint_path}")

    def _fine_tune_with_gradient(self, training_text):
        """Fine-tunes using gradient updates on the full model."""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        loss_fn = torch.nn.CrossEntropyLoss()
        inputs = self.tokenizer(training_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        labels = inputs["input_ids"].clone()

        self.optimizer.zero_grad()
        outputs = self.model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        self.optimizer.step()

        logging.info(f"Gradient-based fine-tuning completed with loss: {loss.item()}")
        self.save_fine_tuned_model()

    def save_fine_tuned_model(self):
        """Saves the fine-tuned model correctly, handling LoRA separately and using automatic device selection."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        model_path = os.path.join(self.config.MODEL_SAVE_DIR, f"{self.config.MODEL_NAME}_{timestamp}")
        os.makedirs(model_path, exist_ok=True)

        logging.info(f"Saving fine-tuned model at {model_path}")

        if self.fine_tune_method == 'lora':
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            self.model.save_adapter(model_path, "default")
            logging.info(f"Saved only LoRA adapters at {model_path}")

        elif self.fine_tune_method == 'gradient':
            # ✅ Save optimizer state only (NO MODEL), with FP16 gradients & pickle protocol 4
            gradient_checkpoint = os.path.join(model_path, "gradient_checkpoint.pt")
            checkpoint = {
                "optimizer_state_dict": {k: v.half().to("cpu") for k, v in self.optimizer.state_dict().items()}  # ✅ Move to CPU before saving
            }
            torch.save(checkpoint, gradient_checkpoint, pickle_protocol=4)
            logging.info(f"Saved optimizer state (FP16 gradients, pickle protocol 4) at {gradient_checkpoint}")

        else:
            self.model.save_pretrained(model_path)
            self.tokenizer.save_pretrained(model_path)
            logging.info(f"Saved full model at {model_path}")

        # ✅ Save to "latest" directory
        latest_model_path = os.path.join(self.config.MODEL_SAVE_DIR, "latest")
        os.makedirs(latest_model_path, exist_ok=True)

        if self.fine_tune_method == 'lora':
            self.model.save_adapter(latest_model_path, "default")
        elif self.fine_tune_method == 'gradient':
            torch.save(checkpoint, os.path.join(latest_model_path, "gradient_checkpoint.pt"), pickle_protocol=4)
        else:
            self.model.save_pretrained(latest_model_path)
            self.tokenizer.save_pretrained(latest_model_path)

        self.db.save_fine_tuning_log(self.config.MODEL_NAME, model_path, "STaR pipeline")
        logging.info(f"Fine-tuned model updated at {model_path} and latest at {latest_model_path}")
