"""
This file aims to fetch the data required to make a
NLP application with disaster tweets
"""
import subprocess
import json
import logging
from dotenv import load_dotenv
import os

load_dotenv()

def setup_logging():
    """
    Set up the logger to display messages on the console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    c_format = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt='%d-%m-%Y %H:%M:%S')
    c_handler.setFormatter(c_format)
    logger.handlers = [c_handler]
    return logger

def fetch_and_organize_data(logger) -> None:
    """
    Download dataset files and organize them in 'dataset' directory.

    Args:
        logger: Logger object.
    """
    try:
        # Download of the dataset (train and test)
        train_file = 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv'
        test_file = 'https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/test.csv'
        subprocess.run(['wget', train_file], check=True)
        subprocess.run(['wget', test_file], check=True)
        subprocess.run(['mkdir', 'dataset'], check=True)

        # Organizing the files in 'dataset' directory
        subprocess.run(['cp', 'train.csv', 'dataset/'], check=True)
        subprocess.run(['cp', 'test.csv', 'dataset/'], check=True)
        logger.info('✅ Fetch data complete!')

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error fetching and organizing data: {e}")

def store_fetch_data(logger) -> None:
    """
    Login to WandB and store the dataset as a WandB artifact.

    Args:
        logger: Logger object.
    """
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        logger.error("❌ API key not found.")
        return

    try:
        # Login in WandB
        subprocess.run(['wandb', 'login', '--relogin', api_key], check=True)
        # Store the dataset as an artifact
        subprocess.run(['wandb', 'artifact', 'put',
                        '--name', 'disaster_tweet_classification/dataset',
                        '--type', 'RawData',
                        '--description', 'Natural Language Processing with Disaster Tweets Dataset',
                        'dataset'], check=True)
        logger.info('✅ Fetch data artifact created with success!')
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error storing and fetching data with WandB: {e}")