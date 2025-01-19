# Importing required libraries
import numpy as np
import pandas as pd
import logging
import os
import sys
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
scaler = StandardScaler()

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = get_project_dir("settings.json") #os.getenv('CONF_PATH')

# Load configuration settings from JSON
logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
INFERENCE_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

# Function to load and split the Iris dataset
def load_and_split_iris(save_train_path: os.path, save_inference_path: os.path, test_size: float = 0.2):
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        test_size=test_size, random_state=42)
    # Standard scaler to train and test X
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Saving training data
    train_df = pd.DataFrame(X_train, columns=iris.feature_names)
    train_df['target'] = y_train
    train_df.to_csv(save_train_path, index=False)
    logger.info(f"Training data saved to {save_train_path}")

    # Saving inference data
    inference_df = pd.DataFrame(X_test, columns=iris.feature_names)
    inference_df['target'] = y_test  # Include target if labels are needed for inference, else drop it
    inference_df.to_csv(save_inference_path, index=False)
    logger.info(f"Inference data saved to {save_inference_path}")

# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    load_and_split_iris(save_train_path=TRAIN_PATH, save_inference_path=INFERENCE_PATH)
    logger.info("Script completed successfully.")
