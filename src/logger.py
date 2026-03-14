import logging
import os
from datetime import datetime

# Setup logging configuration
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'predictions.log')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_prediction(input_data, prediction, probability):
    """
    Registry for auditing model decisions.
    """
    log_msg = f"INPUT: {input_data} | PREDICTION: {prediction} | PROBABILITY: {probability:.4f}"
    logging.info(log_msg)
    print(f"Prediction logged to {log_file}")
