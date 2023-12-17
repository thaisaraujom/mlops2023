"""
Module for segregating the preprocessed dataset into train, 
validation, and test sets.
"""
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow # pylint: disable=import-error
from mlflow import MlflowClient # pylint: disable=import-error
import joblib

def setup_logging() -> tuple:
    """
    Set up logging to a file.

    Returns:
        tuple: A tuple containing the logger, file handler, and log file path.
    """
    logger = logging.getLogger('data_segregation')
    logger.setLevel(logging.INFO)
    log_file = "./logs/data_segregation.log"
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, file_handler, log_file


def download_preprocessed_data() -> str:
    """
    Download the preprocessed dataset from MLflow.

    Returns:
        str: Path to the preprocessed dataset.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    client = MlflowClient()
    experiment_name = "TwitterSentimentAnalysis"
    experiment = client.get_experiment_by_name(experiment_name)

    run_id = None
    for run in client.search_runs(experiment.experiment_id):
        if run.data.tags.get('mlflow.runName') == "preprocessing":
            run_id = run.info.run_id
            break

    if run_id:
        data_dir = "./processed_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        client.download_artifacts(run_id, "preprocessed_twitter_sentiment.csv", data_dir)
        return os.path.join(data_dir, "preprocessed_twitter_sentiment.csv")
    else:
        raise Exception("Preprocessed data not found in MLflow.")


def data_segregation() -> None:
    """
    Segregate the dataset into train, validation, and test sets.
    """
    logger, file_handler, log_file = setup_logging()
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("TwitterSentimentAnalysis")

    with mlflow.start_run(run_name="data_segregation"):
        logger.info("📥 Downloading the preprocessed dataset...")
        dataset_path = download_preprocessed_data()

        logger.info("📜 Preprocessed dataset downloaded successfully.")
        df = pd.read_csv(dataset_path)

        logger.info("🔪 Splitting the dataset into train, validation, and test sets...")
        X = df['clean_text']
        y = df['sentiment'].map({-1: 0, 0: 1, 1: 2, 2: 3})

        (train_x, test_x, train_y, test_y) = train_test_split(X, y, test_size=0.2, random_state=42)
        (train_x, val_x, train_y, val_y) = train_test_split(train_x, train_y,
                                                            test_size=0.2,
                                                            random_state=42)

        logger.info("🚂 Train set size: Features - %s, Labels - %s", len(train_x), len(train_y))
        logger.info("✅ Validation set size: Features - %s, Labels - %s", len(val_x), len(val_y))
        logger.info("🧪 Test set size: Features - %s, Labels - %s", len(test_x), len(test_y))


        logger.info("💾 Saving segregated datasets...")

        # Save datasets using joblib
        joblib.dump(train_x, 'train_x.joblib')
        joblib.dump(train_y, 'train_y.joblib')
        joblib.dump(val_x, 'val_x.joblib')
        joblib.dump(val_y, 'val_y.joblib')
        joblib.dump(test_x, 'test_x.joblib')
        joblib.dump(test_y, 'test_y.joblib')

        # Log datasets as artifacts
        mlflow.log_artifact('train_x.joblib')
        mlflow.log_artifact('train_y.joblib')
        mlflow.log_artifact('val_x.joblib')
        mlflow.log_artifact('val_y.joblib')
        mlflow.log_artifact('test_x.joblib')
        mlflow.log_artifact('test_y.joblib')

        logger.info("🏁 Segregation and saving of datasets completed.")

        # Close the file handler and log the log file as an artifact
        file_handler.close()
        logger.removeHandler(file_handler)
        mlflow.log_artifact(log_file)
