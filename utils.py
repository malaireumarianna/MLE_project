import os
import logging
import logger
from loguru import logger
import sys

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

def get_project_dir(sub_dir: str) -> str:
    """Return path to a project subdirectory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), sub_dir))

def configure_logger() -> None:
    """Configures logging"""
    '''handlers = [
        logging.FileHandler("training.log"),  # Logs to a file
        logging.StreamHandler(sys.stdout)  # Logs to the console
    ]

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=handlers)'''


    '''logging.basicConfig(filename='training.log',
                            level=logging.INFO,
                            filemode='w',  # Overwrites the file each time the application starts
                            format='%(asctime)s - %(levelname)s - %(message)s')'''

    logger.add("_logs_for_train.txt", level="INFO")

    # Log a test message to ensure logging is working
    logger.info("Logging setup complete. Starting training...")
