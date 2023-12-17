"""
This module contains the functions for preprocessing the dataset.
"""
import re
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import mlflow # pylint: disable=import-error
from mlflow import MlflowClient

# Ensure that the necessary NLTK downloads are present
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Set up logging to a file
def setup_logging() -> tuple:
    """
    Set up logging to a file.
    """
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.INFO)
    log_file = "./logs/preprocessing.log"
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
        raise Exception(f"The experiment '{experiment_name}' does not exist.")

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
        raise LookupError(
            f"No runs found with name 'fetch_data' "
            f"for experiment '{experiment_name}'."
        )



def lowercasing(text):
    """
    Converts all characters of the input text to lowercase.

    Args:
        text: Input text string.

    Returns:
        Lowercased string of text.
    """
    return text.lower()

def punctuations(text):
    """
    Removes non-alphabetic characters from the input text.
    """
    return re.sub(r"[^a-zA-Z]", " ", text)

def tokenization(text):
    """
    Tokenizes the input text into individual words.
    """
    return word_tokenize(text)

def stopwords_remove(tokens):
    """
    Removes common English stopwords from a list of tokens.
    """
    stop_words = set(stopwords.words("english"))
    return [word for word in tokens if word not in stop_words]

def lemmatization(tokens):
    """
    Performs lemmatization on a list of tokens using WordNet lemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word=word, pos="v") for word in tokens]

def remove_urls_and_handles(text):
    """
    Removes URLs and Twitter handles from the input text.

    Args:
        text: Input text string.

    Returns:
        String with URLs and Twitter handles removed.
    """
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove Twitter handles
    return text

def remove_rt(text):
    """
    Removes the retweet abbreviation "rt" from the input text.

    Args:
        text: Input text string.

    Returns:
        String with "rt" removed.
    """
    return re.sub(r'\brt\b', '', text)

def text_preprocessing(df):
    """
    Performs text preprocessing on the 'message' column of the given DataFrame.
    """
    # Convert to lower case
    df['clean_text'] = df['message'].apply(lowercasing)

    # Remove URLs and Twitter handles
    df['clean_text'] = df['clean_text'].apply(remove_urls_and_handles)

    # Remove 'RT' for retweet
    df['clean_text'] = df['clean_text'].apply(remove_rt)

    # Remove punctuations
    df['clean_text'] = df['clean_text'].apply(punctuations)

    # Tokenize text
    df['tokens'] = df['clean_text'].apply(tokenization)

    # Remove stopwords
    df['tokens'] = df['tokens'].apply(stopwords_remove)

    # Perform lemmatization
    df['tokens'] = df['tokens'].apply(lemmatization)

    return df

def create_word_clouds(df, sentiment_column, text_column, logger):
    """
    Creates and logs word clouds for each unique sentiment in the DataFrame.

    Args:
        df: pandas DataFrame containing the tweet data.
        sentiment_column: the name of the column in df that contains the sentiment labels.
        text_column: the name of the column in df that contains the text of the tweets.
    """
    logger.info("🌥 Creating word clouds for each sentiment...")

    sentiments = df[sentiment_column].unique()
    for sentiment in sentiments:
        subset = df[df[sentiment_column] == sentiment]
        text = " ".join(review for review in subset[text_column])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Define the file path for saving the word cloud image
        wordcloud_file = f"./logs/wordcloud_sentiment_{sentiment}.png"

        # Generate the word cloud image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for Sentiment {sentiment}')
        plt.axis('off')

        # Save the word cloud image to the file system
        plt.savefig(wordcloud_file)
        plt.close()

        # Log the word cloud image as an artifact
        mlflow.log_artifact(wordcloud_file)
        logger.info(f"🎨 Word cloud for sentiment {sentiment} created and logged as an artifact.")

def preprocessing() -> None:
    """
    Main function for preprocessing the dataset.
    """
    try:
        logger, file_handler, log_file = setup_logging()
        logger.info("🚀 Starting preprocessing...")
        download_artifact()
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        mlflow.set_experiment("TwitterSentimentAnalysis")

        # Start MLflow run for preprocessing
        with mlflow.start_run(run_name="preprocessing"):
            # Load your dataset
            dataset_path = "./data/twitter_sentiment.csv"
            df = pd.read_csv(dataset_path)

            # Perform preprocessing
            df = text_preprocessing(df)
            logger.info("📝 Text preprocessing completed.")

            # Save the preprocessed dataset as a new artifact
            preprocessed_file = "preprocessed_twitter_sentiment.csv"
            df.to_csv(preprocessed_file, index=False)
            mlflow.log_artifact(preprocessed_file)
            logger.info("📥 Preprocessed dataset saved and logged as an artifact.")

            # Create and log word clouds
            create_word_clouds(df, 'sentiment', 'clean_text', logger)

            # Log the log file as an artifact
            file_handler.close()
            logger.removeHandler(file_handler)
            mlflow.log_artifact(log_file)

    except Exception as e:
        logger.error("An error occurred during preprocessing: %s", e)
        raise
