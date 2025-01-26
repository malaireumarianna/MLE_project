"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""
import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger


# Comment this line if you have problems with MLFlow installation
import mlflow
mlflow.pytorch.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logger
# Change to CONF_FILE = "settings.json" if you have problems with env variables
#CONF_FILE = os.getenv('CONF_PATH')
CONF_FILE = get_project_dir("settings.json")

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(4, 256)  # Increase the number of neurons
        # Additional hidden layers
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)  # New layer
        self.fc5 = nn.Linear(32, 3)   # Output layer

        # Dropout layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% probability

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first hidden layer
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # Output layer with logits for 3 classes
        return x


def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '_best_model.pth') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '_best_model.pth'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str) -> IrisNN:
    """Loads and returns the specified model"""
    try:
        model = IrisNN()
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f'Model loaded from {path}')
        return model
    except Exception as e:
        logger.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)


def get_inference_data(path: str, batch_size: int = 16) -> DataLoader:
    """loads and returns data for inference """
    try:
        df = pd.read_csv(path)

        features = df.drop('target', axis=1, errors='ignore').values.astype(np.float32)
        #errors='ignore'  if the 'target' column does not exist

        if 'target' in df.columns:

            labels = df['target'].values.astype(np.int64)
            val_data = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
            # Create Data Loader
        else:
            val_data = TensorDataset(torch.from_numpy(features))
        # Create Data Loader
        val_loader = DataLoader(val_data, batch_size=batch_size)

        return val_loader

    except Exception as e:
        logger.error(f"An error occurred while loading inference data: {e}")
        sys.exit(1)


def predict_results(model: IrisNN, criterion: nn.CrossEntropyLoss(), infer_data: DataLoader) -> pd.DataFrame:
    """Predict de results and join it with the infer_data"""

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    all_val_labels = []
    all_val_predictions = []
    with torch.no_grad():
        for batch in infer_data:
            print(len(batch))  # Debug: Check the batch structure
            if len(batch) == 2: #isinstance(batch, tuple) and
                data, target = batch
            else:
                data = batch[0]
                target = None

            output = model(data)

            if target != None:
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
                all_val_labels.append(target.cpu().numpy())
            else:
                predicted = torch.max(output, 1)[1]  # Get indices of max log-probability

            all_val_predictions.append(predicted.cpu().numpy())

    # Calculate accuracy if targets are available
    if total_val > 0:
        val_accuracy = 100 * correct_val / total_val
        logger.info(f'Val Loss: {val_loss / len(infer_data):.4f}, Val Accuracy: {val_accuracy:.2f}%')
    else:
        logger.info('No target labels available for accuracy computation.')

    # Prepare results DataFrame
    print(all_val_labels)
    if all_val_labels:
        results_df = pd.DataFrame({
            'Actual': np.concatenate(all_val_labels),
            'Predicted': np.concatenate(all_val_predictions)
        })
    else:
        results_df = pd.DataFrame({
            'Predicted': np.concatenate(all_val_predictions)
        })

    return results_df


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logger.info(f'Results saved to {path}')


def main():
    """Main function"""
    logger.add("_logs_for_test.txt", level="INFO")

    # Log a test message to ensure logger is working
    logger.info("logger setup complete. Starting testing...")


    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model,  nn.CrossEntropyLoss(), infer_data)
    store_results(results, args.out_path)

    logger.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()