"""
This module contains the application interface.
"""

import logging
import sys
from transformers import AutoTokenizer
import tensorflow as tf
import gradio as gr # pylint: disable=import-error
import pandas as pd
import numpy as np
from src.iii_preprocessing import (
    tokenization, stopwords_remove, lemmatization
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_text(text: str) -> pd.Series:
    """
    Performs text preprocessing.

    Args:
        text (str): Input text string.

    Returns:
        pd.Series: Preprocessed text.
    """
    lower_text = text.lower()
    tokenized_text = tokenization(lower_text)
    no_stopwords = stopwords_remove(tokenized_text)
    lemmatized_text = lemmatization(no_stopwords)
    joined_text = ' '.join(lemmatized_text)
    return pd.Series(joined_text)

def make_prediction(preprocessed_text: pd.Series,
                    model, tokenizer) -> str:
    """
    Makes a prediction based on preprocessed text.

    Args:
        preprocessed_text (pd.Series): Preprocessed text.
        model: Trained model.
        tokenizer: Tokenizer object.

    Returns:
        str: Sentiment label.
    """
    tokenized = tokenizer(list(preprocessed_text), truncation=True, padding=True)
    outputs = model(tokenized)
    classification = np.argmax(outputs['logits'], axis=1)

    label = {0: 'Anti', 1: 'Neutral', 2: 'Pro', 3: 'News'}

    classification = label[classification[0]]

    return classification

def classify_tweet(tweet: str, model, tokenizer) -> str:
    """
    Classifies the sentiment of a tweet.

    Args:
        tweet (str): Tweet text.
        model: Trained model.
        tokenizer: Tokenizer object.

    Returns:
        str: Sentiment label.
    """
    preprocessed = preprocess_text(tweet)
    return str(make_prediction(preprocessed, model, tokenizer))

if __name__ == "__main__":
    try:
        bert_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        MODEL_PATH = "./model_data/model/data/model"
        sentiment_model = tf.keras.models.load_model(MODEL_PATH)
    except (FileNotFoundError, tf.errors.NotFoundError,
            tf.errors.OpError) as e:
        logging.error("Error loading model or tokenizer: %s", e)
        sys.exit()

    gr_interface = gr.Interface(
        title="Climate Change Sentiment",
        description=(
            "Classify your feelings about climate change. "
            "Classifications: \n"
            "- `News`: Factual news; \n"
            "- `Pro`: Supports man-made climate change belief; \n"
            "- `Neutral`: Neutral; \n"
            "- `Anti`: Does not believe in it.\n"
        ),
        article=(
            "Developed by **Mariana Azevedo** and **Thaís Medeiros** "
            "for the final work of the Machine Learning-Based Systems"
            " Project course, taught by professor Ivanovitch Silva at UFRN."
        ),
        fn=lambda tweet: classify_tweet(tweet, sentiment_model, bert_tokenizer),
        inputs=gr.Textbox(
            lines=2,
            placeholder="What do you think about climate change?",
            label='text'
        ),
        outputs='text'
    )

    gr_interface.launch()
