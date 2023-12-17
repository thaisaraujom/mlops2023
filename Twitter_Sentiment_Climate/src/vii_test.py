"""
This module contains the code for testing the model.
"""
import os
import logging
import mlflow # pylint: disable=import-error
import mlflow.tensorflow # pylint: disable=import-error
import joblib
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from mlflow import MlflowClient # pylint: disable=import-error

def setup_logging() -> tuple:
    """
    Set up logging to a file.

    Returns:
        tuple: A tuple containing the logger, file handler, 
        and log file path.
    """
    logger = logging.getLogger('test_model')
    logger.setLevel(logging.INFO)
    log_file = "./logs/test_model.log"
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, file_handler, log_file

def download_test_data() -> list:
    """
    Download the test data from a specified MLflow run.

    Returns:
        list: A list containing the paths to the downloaded artifacts.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    client = mlflow.tracking.MlflowClient()
    run_name='data_segregation'
    artifact_names=['test_x.joblib', 'test_y.joblib']
    # Find the run ID of the specified run name
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name("TwitterSentimentAnalysis").experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'"
    )
    if not runs:
        raise Exception(f"No runs found for '{run_name}'")

    run_id = runs[0].info.run_id

    # Download the artifacts
    local_path = "local_test_data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    downloaded_files = []
    for artifact_name in artifact_names:
        client.download_artifacts(run_id, artifact_name, local_path)
        downloaded_files.append(os.path.join(local_path, artifact_name))

    return downloaded_files


def download_model(logger) -> str:
    """
    Download the model from MLflow.

    Args:
        logger: Logger object.
    
    Returns:
        str: Path to the directory containing the MLmodel.
    """
    run_name = "train_model"
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("TwitterSentimentAnalysis")
    if not experiment:
        raise Exception(f"No experiment found with name 'TwitterSentimentAnalysis'")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              filter_string=f"tags.mlflow.runName='{run_name}'")
    if not runs:
        raise Exception(
            f"No runs found with run name '{run_name}' in experiment 'TwitterSentimentAnalysis'"
            )

    run_id = runs[0].info.run_id

    # Path to the directory where the model will be downloaded
    model_dir = "./model_data"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # The artifact path should point to the 'model' directory
    artifact_path = "model"

    # Download the 'model' directory artifact from the run
    local_model_dir_path = client.download_artifacts(run_id, artifact_path, model_dir)
    if not local_model_dir_path:
        raise Exception(f"Failed to download model for run_id: {run_id}")

    logger.info(f"✅ Model from run {run_name} downloaded successfully to {local_model_dir_path}.")
    return local_model_dir_path  # Return the path to the directory containing the MLmodel


def load_test_data() -> tuple:
    """
    Load the test data.

    Returns:
        tuple: A tuple containing the test data and labels.
    """
    test_x_path, test_y_path = download_test_data()
    test_x = joblib.load(test_x_path)
    test_y = joblib.load(test_y_path)
    return test_x, test_y

def evaluate_model(model, test_x, test_y) -> tuple:
    """
    Evaluate the model on the test data.

    Args:
        model: Trained model.
        test_x: Test data.
        test_y: Test labels.

    Returns:
        tuple: A tuple containing the classification report and 
        the path to the confusion matrix.
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized = tokenizer(list(test_x), truncation=True, padding=True)
    outputs = model(tokenized)
    classifications = np.argmax(outputs['logits'], axis=1)

    # Classification report
    report = classification_report(test_y, classifications)

    # Confusion matrix
    cm = confusion_matrix(test_y, classifications)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ConfusionMatrixDisplay(cm).plot(values_format=".0f", ax=ax)
    ax.set_xlabel("True Label")
    ax.set_ylabel("Predicted Label")
    ax.grid(False)
    plt.savefig("confusion_matrix.png")
    plt.close(fig)

    return report, "confusion_matrix.png"

def test_model() -> None:
    """
    Test the model.
    """
    logger, file_handler, log_file = setup_logging()

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("TwitterSentimentAnalysis")

    with mlflow.start_run(run_name="test_model"):
        model_path = download_model(logger)
        model = mlflow.tensorflow.load_model(model_path)

        logger.info("📚 Loading test data...")
        test_x, test_y = load_test_data()

        logger.info("🔍 Evaluating model...")
        report, confusion_matrix_path = evaluate_model(model, test_x, test_y)
        logger.info("Classification Report:\n%s", report)

        file_handler.close()
        logger.removeHandler(file_handler)
        mlflow.log_artifact(log_file)
        mlflow.log_artifact(confusion_matrix_path)
