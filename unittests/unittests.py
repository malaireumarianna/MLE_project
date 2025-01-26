import pandas as pd
import os
import sys
import json
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = os.getenv('CONF_PATH')
from utils import get_project_dir
CONF_FILE = get_project_dir("settings.json")

from training.train import DataProcessor, Training 
from inference.run import IrisNN, predict_results


class TestEvaluateModel(unittest.TestCase):

    def setUp(self):

        self.model = IrisNN()
        self.criterion = torch.nn.CrossEntropyLoss()


        self.features = torch.randn(20, 10)
        test= pd.DataFrame({
            'x1': [1, 0, 1, 0],
            'x2': [1, 1, 0, 0]
        }).values.astype(np.float32)
        self.features = torch.from_numpy(test)

        self.features = torch.randn(20, 4)  
        self.labels = torch.randint(0, 2, (20,))

    def test_evaluate_with_targets(self):
        
        dataset_with_targets = TensorDataset(self.features, self.labels)
        data_loader_with_targets = DataLoader(dataset_with_targets, batch_size=4)

        results_df = predict_results(self.model, self.criterion, data_loader_with_targets)


        # Check that results DataFrame contains 'Actual' and 'Predicted' columns
        self.assertIn('Actual', results_df.columns)
        self.assertIn('Predicted', results_df.columns)
        self.assertEqual(len(results_df['Actual']), 20)  
        self.assertEqual(len(results_df['Predicted']), 20)

    def test_evaluate_without_targets(self):
        
        dataset_without_targets = TensorDataset(self.features)
        data_loader_without_targets = DataLoader(dataset_without_targets, batch_size=4)

        results_df = predict_results(self.model, self.criterion, data_loader_without_targets)

        
        self.assertNotIn('Actual', results_df.columns)
        self.assertIn('Predicted', results_df.columns)
        self.assertEqual(len(results_df['Predicted']), 20)  


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(100)
        self.assertEqual(df.shape[0], 100)


class TestTraining(unittest.TestCase):
    def test_train(self):
        tr = Training(IrisNN())
        
        X_train = pd.DataFrame({
            'x1': [1, 0, 1, 0],
            'x2': [1, 1, 0, 0],
            'x3': [0, 1, 0, 1],  
            'x4': [0, 0, 1, 1],
            'target': [0, 1, 1, 0]
        })

        tr.run_training(X_train, test_size=0.25)

        self.assertIsNotNone(tr.model)

if __name__ == '__main__':
    unittest.main()
