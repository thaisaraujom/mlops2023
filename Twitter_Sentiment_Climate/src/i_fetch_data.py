"""
This module fetches the dataset from Kaggle and logs the process.
"""
import logging
import mlflow # pylint: disable=import-error

def fetch_data() -> None:
    """
    Fetches the dataset from Kaggle and logs the process.
    """
    # Setting up the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_file = "./logs/fetch_data.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("🚀 Starting data fetch process...")

    # MLflow configuration
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("TwitterSentimentAnalysis")

    # Starting a run in MLflow
    with mlflow.start_run(run_name="fetch_data"):
        logger.info("✅ MLflow tracking URI set and experiment started.")

        # Specify the path to the dataset
        dataset_path = "./data/twitter_sentiment.csv"
        logger.info("📁 Dataset path set to: %s", dataset_path)

        # Log the dataset file as an artifact
        mlflow.log_artifact(dataset_path)
        logger.info("📤 Logging the dataset artifact: %s", dataset_path)

        # Log the log file as an artifact
        file_handler.close()
        logger.removeHandler(file_handler)
        mlflow.log_artifact(log_file)
