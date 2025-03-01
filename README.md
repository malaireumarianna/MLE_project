
## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic_example
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── inference_data.csv
│   └── train_data.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── unittests                  # Scripts used for unittests
│   └── unittests.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file or manually initialize an environment variable as `CONF_PATH=settings.json`.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.

## Data:
Data is the cornerstone of any Machine Learning project. For generating the data, use the script located at `data_process/data_generation.py`. The generated data is used to train the model and to test the inference. Following the approach of separating concerns, the responsibility of data generation lies with this script.

## Training:
The training phase of the ML pipeline includes preprocessing of data, the actual training of the model, and the evaluation and validation of the model's performance. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -it training_image /bin/bash
```
In this directory can be found file `_logs_for_train.txt` which contains detailed logs for training process, accuracy and loss for testing the model on subset of train data.

- Enter the repository with models and copy the name of saved model by using commands below:
```bash
cd models
```
```bash
ls
```
- Then exit to be able to run copying of model by docker command:
```bash
exit
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:

```bash
docker cp <container_id>:/app/models/<model_name>.pth ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pth` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pth --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -it inference_image /bin/bash  
```
In this directory can be found file `_logs_for_test.txt` which contains detailed logs for inference.

After that ensure that you have your results in the `results` directory in your inference container. It should contain csv file with the Predicted values.


2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.

## Unittesting

1. TestEvaluateModel
This test class is designed to evaluate the performance of a model on datasets with and without target labels. It uses a neural network model (IrisNN), which is used for the Iris dataset. It tests scripts from inference/run.py, which contains script for inference without target variable or in case of its presence, like in ours it will calculate loss and accuracy.
It contains two tests - test_evaluate_with_targets and test_evaluate_without_targets.

2. TestDataProcessor
This class tests the functionality of a DataProcessor class from train.py that is used for preparing data. This checks whether the data is trimmed to a specified size correctly.

3. TestTraining
This test class checks the training process of a machine learning model.
