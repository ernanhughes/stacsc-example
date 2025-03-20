import logging
from tqdm import tqdm

from star import STaR


star_pipeline = STaR()
questions = [
    "How does a neural network backpropagation work?",
    "What are the key differences between supervised and unsupervised learning?",
    "Explain the concept of reinforcement learning with examples.",
    "How do decision trees handle missing values?",
    "Describe the process of hyperparameter tuning in machine learning.",
    "What are the advantages of using transformers over LSTMs?",
    "Explain the role of activation functions in deep learning.",
    "How do generative adversarial networks (GANs) work?",
    "What is transfer learning and why is it useful?",
    "Describe the differences between batch normalization and layer normalization."
]

for question in tqdm(questions, desc="Processing questions", unit="question", leave=True, position=0):
    logging.info(f"Processing question: {question}")
    paths = [question]  # Placeholder, replace with actual model outputs
    best_answer = star_pipeline.evaluate_paths(question, paths)
    logging.info(f"Best answer for '{question}': {best_answer}")
