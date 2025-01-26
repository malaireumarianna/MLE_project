import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loguru import logger


# Comment this line if you have problems with MLFlow installation
import mlflow
mlflow.pytorch.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logger

# Loads configuration settings from JSON
#CONF_FILE = os.getenv('CONF_PATH', os.path.join(ROOT_DIR, 'settings.json'))


CONF_FILE = get_project_dir("settings.json") #os.getenv('CONF_PATH')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")

class DataProcessor:
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logger.info("Preparing data for training...")

        df = pd.read_csv(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if max_rows and max_rows > 0 and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=conf['general']['random_state'])
            logger.info(f'Random sampling performed. Sample size: {max_rows}')
        else:
            logger.info('Skipping sampling.')

        return df



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


class Training:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.patience = 25 # Number of epochs to wait before stopping if no improvement
        self.best_val_loss = float('inf')  # Initialize best validation loss to a very large value
        self.early_stopping_counter = 0  # Counter to track epochs without improvement
        self.best_model_path = os.path.join(MODEL_DIR, datetime.now().strftime(
            conf['general']['datetime_format']) + '_best_model.pth')

    def run_training(self, df: pd.DataFrame, test_size: float = 0.2, batch_size: int = 16):
        logger.info("Running training...")
        features = df.drop('target', axis=1).values.astype(np.float32)
        labels = df['target'].values.astype(np.int64)
        #features = self.scaler.fit_transform(features)

        X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=test_size, random_state=conf['general']['random_state'])

        # Convert to PyTorch tensors
        train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

        # Create Data Loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)


        # Early stopping and learning rate scheduler setup

        lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # Training loop
        for epoch in range(100):  # Number of epochs
            self.model.train()
            running_loss = 0.0
            for data, target in train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validation loop
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total_val += target.size(0)
                    correct_val += (predicted == target).sum().item()

            val_accuracy = 100 * correct_val / total_val

            # Learning rate scheduler and early stopping check
            lr_scheduler.step(val_loss)

            # Check for improvement in validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # Reset the counter
                self.save()  # Save the best model
                logger.info(f'Epoch {epoch + 1}: Validation loss improved. Model saved to {self.best_model_path}')
            else:
                self.early_stopping_counter += 1
                logger.info(
                    f'Epoch {epoch + 1}: No improvement in validation loss. Early stopping counter: {self.early_stopping_counter}/{self.patience}')

            # Early stopping
            if self.early_stopping_counter >= self.patience:
                logger.info(
                    f'Early stopping triggered after {epoch + 1} epochs. Best validation loss: {self.best_val_loss:.6f}')
                break
            logger.info(f"Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

            logger.info(
                f'Epoch {epoch + 1}, Train Loss: {running_loss / len(train_loader):.4f},  Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')



    def save(self,) -> None:
        logger.info("Saving the model...")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        torch.save(self.model.state_dict(), self.best_model_path)

def main():
    logger.add("_logs_for_train.txt", level="INFO")

    # Log a test message to ensure logging is working
    logger.info("Logging setup complete. Starting training...")


    data_proc = DataProcessor()
    model = IrisNN()
    tr = Training(model)

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])
    #tr.save(datetime.now().strftime(conf['general']['datetime_format']) + '_model.pth')

if __name__ == "__main__":
    main()
