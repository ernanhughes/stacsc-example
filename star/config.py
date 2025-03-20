import torch
import sqlite3
import logging
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    filename="star_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Config:
    CONFIDENCE_THRESHOLD = 0.7  # Default threshold for correctness determination
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    DB_NAME = "reasoning_paths.db"
    LOG_FILE = "star_pipeline.log"
    NUM_PATHS = 3
    MAX_LENGTH = 100
    LEARNING_RATE = 5e-5
    EPOCHS = 3
    QUESTIONS_FILE = "questions.txt"
    MODEL_SAVE_DIR = "fine_tuned_models"
    CONFIDENCE_METHOD = "model"  # Options: "model", "heuristic", "external"
