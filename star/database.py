import sqlite3
from config import Config

import logging
import torch

class Database:
    def __init__(self, db_name=Config.DB_NAME):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            reasoning_path TEXT,
            is_correct BOOLEAN,
            confidence REAL DEFAULT 0.0,
            is_best BOOLEAN DEFAULT 0
        )""")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fine_tuning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_model TEXT,
            model_path TEXT,
            source TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        conn.close()
        logging.info("Database initialized.")

    def save_fine_tuning_log(self, base_model, model_path, source):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO fine_tuning_log (base_model, model_path, source) VALUES (?, ?, ?)",
                       (base_model, model_path, source))
        conn.commit()
        conn.close()
        logging.info(f"Fine-tuning logged: {model_path} from {source}")

    def get_latest_fine_tuned_model(self, model_name=Config.MODEL_NAME):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT model_path FROM fine_tuning_log WHERE base_model = ? ORDER BY timestamp DESC LIMIT 1",
                       (model_name,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        else:
            return None

    def save_path(self, question, path, is_correct, confidence):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO paths (question, reasoning_path, is_correct, confidence, is_best) VALUES (?, ?, ?, ?, ?)",
            (question, path, is_correct, confidence, 0))
        conn.commit()
        conn.close()
        logging.info(f"Saved path for question: {question} with confidence: {confidence}")

    def update_best_path(self, question, best_path):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        # First, reset all paths for the question to not be best
        cursor.execute("UPDATE paths SET is_best = 0 WHERE question = ?", (question,))
        # Then, set the best path
        cursor.execute("UPDATE paths SET is_best = 1 WHERE question = ? AND reasoning_path = ?", (question, best_path))
        conn.commit()
        conn.close()
        logging.info(f"Updated best path for question: {question}")


