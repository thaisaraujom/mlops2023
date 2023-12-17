"""
This module contains the code for Exploratory Data Analysis (EDA) of 
the Twitter Sentiment Analysis dataset.
"""
import os
import logging
import mlflow # pylint: disable=import-error
from mlflow import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt

# Setting up the logger
def setup_logging() -> tuple:
    """
    Set up logging to a file.

    Returns:
        tuple: A tuple containing the logger, file handler, 
        and log file path.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_file = "./logs/eda.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger, file_handler, log_file

def download_artifact() -> None:
    """
    Downloads the dataset artifact from MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    client = MlflowClient()

    # Hardcoded experiment name
    experiment_name = "TwitterSentimentAnalysis"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"The experiment '{experiment_name}' does not exist.")

    # List all runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id],
                              order_by=["start_time desc"])

    # Filter runs by run name "fetch_data"
    run_id = None
    for run in runs:
        if run.data.tags.get('mlflow.runName') == "fetch_data":
            run_id = run.info.run_id
            break

    if run_id is not None:
        data_dir = "./data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the artifact
        client.download_artifacts(run_id, "twitter_sentiment.csv", data_dir)
    else:
        raise LookupError(f"No runs found with name 'fetch_data' for experiment '{experiment_name}'.")


def eda() -> None:
    """
    Performs EDA on the Twitter Sentiment Analysis dataset.
    """
    # Setting up the logger
    logger, file_handler, log_file = setup_logging()
    download_artifact()
    logger.info("🚀 Starting EDA process...")

    # MLflow configuration
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("TwitterSentimentAnalysis")

    # Starting a run in MLflow for EDA
    with mlflow.start_run(run_name="eda"):
        # Read the CSV file
        dataset_path = "./data/twitter_sentiment.csv"
        df = pd.read_csv(dataset_path)
        logger.info("✅ Dataset loaded for EDA.")

        # Save the first 5 rows to a CSV and log it
        head_file = "./logs/df_head.csv"
        df.head().to_csv(head_file, index=False)
        mlflow.log_artifact(head_file)

        # Save the null values information to a text file and log it
        null_file = "./logs/df_nulls.txt"
        with open(null_file, 'w') as f:
            f.write(str(df.isnull().sum()))
        mlflow.log_artifact(null_file)

        # Plotting and saving the sentiment distribution
        plot_file = "./logs/sentiment_distribution.png"
        df['sentiment'].value_counts().plot(kind='bar')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.savefig(plot_file)
        mlflow.log_artifact(plot_file)
        logger.info("📊 Sentiment distribution plot saved and logged.")

        # Close the file handler
        file_handler.close()
        logger.removeHandler(file_handler)
        mlflow.log_artifact(log_file)
