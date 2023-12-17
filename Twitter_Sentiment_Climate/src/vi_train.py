"""
This module contains the code for training the model.
"""
import logging
import os
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import mlflow # pylint: disable=import-error
from mlflow import MlflowClient # pylint: disable=import-error
import joblib

# Set up logging
def setup_logging() -> tuple:
    """
    Set up logging to a file.

    Returns:
        tuple: A tuple containing the logger, file handler, and log file path.
    """
    logger = logging.getLogger('train_model')
    logger.setLevel(logging.INFO)
    log_file = "./logs/train_model.log"
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, file_handler, log_file

# Downloading the segregated dataset
def download_segregated_data(logger) -> str:
    """
    Download the segregated dataset from MLflow.

    Args:
        logger: Logger object.
    Returns:
        str: Path to the directory containing the segregated datasets.
    """
    run_name="data_segregation"
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    client = MlflowClient()
    experiment_name = "TwitterSentimentAnalysis"
    experiment = client.get_experiment_by_name(experiment_name)

    run_id = None
    for run in client.search_runs(experiment.experiment_id):
        if run.data.tags.get('mlflow.runName') == run_name:
            run_id = run.info.run_id
            break

    if run_id:
        data_dir = f"./{run_name}_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for file_name in ['train_x.joblib',
                          'train_y.joblib',
                          'val_x.joblib',
                          'val_y.joblib',
                          'test_x.joblib',
                          'test_y.joblib']:
            client.download_artifacts(run_id, file_name, data_dir)
        logger.info(f"✅ Data from {run_name} downloaded successfully.")
        return data_dir
    raise FileNotFoundError(f"{run_name} data not found in MLflow.")


def training(logger) -> None:
    """
    Train the model.

    Args:
        logger: Logger object.
    """
    logger.info("🚀 Starting model training...")

    # Downloading segregated data
    data_dir = download_segregated_data(logger)

    # Load the segregated datasets using joblib
    train_x = joblib.load(os.path.join(data_dir, 'train_x.joblib'))
    train_y = joblib.load(os.path.join(data_dir, 'train_y.joblib'))
    val_x = joblib.load(os.path.join(data_dir, 'val_x.joblib'))
    val_y = joblib.load(os.path.join(data_dir, 'val_y.joblib'))

    logger.info("📜 Segregated datasets loaded.")

    # Tokenize the data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    logger.info("🤖 Tokenizer loaded.")

    train_encodings = tokenizer(list(train_x),
                                truncation=True,
                                padding='max_length',
                                return_tensors='tf')
    val_encodings = tokenizer(list(val_x),
                              truncation=True,
                              padding='max_length',
                              return_tensors='tf')
    logger.info("✨ Data tokenization completed.")

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),
                                                        tf.constant(train_y.values,
                                                                    dtype=tf.int32)))
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings),
                                                      tf.constant(val_y.values,
                                                                  dtype=tf.int32)))
    logger.info("📚 TensorFlow datasets created.")

    train_dataset = train_dataset.batch(8)
    val_dataset = val_dataset.batch(8)
    logger.info("📦 Datasets batched.")

    # Model configuration and training
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                 num_labels=4)

    # pylint: disable=E1101
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    logger.info("🛠️ Model compiled.")

    model.fit(train_dataset, epochs=1, validation_data=val_dataset)
    logger.info("🏋️ Model trained.")

    # Register the model with MLflow
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
    )

    logger.info("✅ TensorFlow model successfully logged to MLflow.")
    logger.info("💾 Model saved and logged in MLflow.")

def train() -> None:
    """
    Main function for training the model.
    """
    # Set up logging
    logger, file_handler, log_file = setup_logging()

    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("TwitterSentimentAnalysis")

    with mlflow.start_run(run_name="train_model"):
        training(logger)

        # Close the file handler and log the log file as an artifact
        file_handler.close()
        logger.removeHandler(file_handler)
        mlflow.log_artifact(log_file)
