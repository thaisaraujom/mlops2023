"""
This file aims to preprocess textual data related to 
disaster tweets and generate illustrative word clouds.
"""
import re
import wandb
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
matplotlib.use('Agg')

def download_dataset_artifact(logger) -> None:
    """
    Initialize a wandb run and download "dataset" artifact.

    Args:
        logger: Logger object.
    """
    try:
        # Initialize wandb run
        run = wandb.init(
            project="disaster_tweet_classification",
            save_code=True,
            job_type="preprocessing",
        )

        # Get the artifact
        artifact = run.use_artifact("dataset:v0")

        # Download the content of the artifact to the local directory
        artifact_dir = artifact.download()

        # Path to dataset
        data_path = artifact_dir + "/train.csv"

        # Load dataset
        df_disaster_tweet = pd.read_csv(data_path)
        logger.info("✅ Artifact downloaded with success!")

        return {"df_disaster_tweet": df_disaster_tweet, "run_id": run.id}
    except FileNotFoundError as fnf_error:
        logger.error(f"❌ File not found error during artifact download: {fnf_error}")
    except pd.errors.ParserError as parse_error:
        logger.error(f"❌ Parsing error during data loading: {parse_error}")
    except wandb.errors.CommError as wandb_error:
        logger.error(f"❌ Wandb communication error: {wandb_error}")
    except Exception as error_eda:
        logger.error(f"❌ Unexpected error during artifact download: {error_eda}")
    return {"df_disaster_tweet": None, "run_id": None}



def punctuations(text: str) -> str:
    """
    Removes non-alphabetic characters from the input text.

    Args:
        text: Input text string.

    Returns:
        String with non-alphabetic characters removed.
    """
    return re.sub(r"[^a-zA-Z]", " ", text)


def tokenization(text: str) -> list:
    """
    Tokenizes the input text into individual words.
    """
    return word_tokenize(text)


def stopwords_remove(tokens: list) -> list:
    """
    Removes common English stopwords from a list of tokens.

    Args:
        tokens: List of tokens.

    Returns:
        List of tokens with no stopwords.
    """
    stop_words = set(stopwords.words("english"))
    stop_words.remove("not")  # Preserve 'not' for sentiment analysis
    return [word for word in tokens if word not in stop_words]


def lemmatization(tokens: list) -> list:
    """
    Performs lemmatization on a list of tokens using WordNet lemmatizer.

    Args:
        tokens: List of tokens.

    Returns:
        List of lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word=word, pos="v") for word in tokens]


def text_preprocessing(df_disaster_tweet, logger, run_id) -> dict or None:
    """
    Performs text preprocessing on the 'text' column of the given DataFrame.

    Args:
        df_disaster_tweet: DataFrame with the dataset.
        logger: Logger object.
        run_id: Wandb run id.
    Returns:
        Dictionary with the preprocessed DataFrame and
        two DataFrames with disaster and non-disaster tweets.
    """
    try:
        run = wandb.init(
            project="disaster_tweet_classification", id=run_id, resume=True
        )
        df_disaster_tweet["text_lower"] = df_disaster_tweet["text"].str.lower()
        df_disaster_tweet["text_no_punct"] = df_disaster_tweet["text_lower"].apply(
            punctuations
        )
        df_disaster_tweet["text_tokenized"] = df_disaster_tweet["text_no_punct"].apply(
            tokenization
        )
        df_disaster_tweet["text_no_stop"] = df_disaster_tweet["text_tokenized"].apply(
            stopwords_remove
        )
        df_disaster_tweet["text_lemmatized"] = df_disaster_tweet["text_no_stop"].apply(
            lemmatization
        )
        df_disaster_tweet["final"] = df_disaster_tweet["text_lemmatized"].apply(
            " ".join
        )

        for column in [
            "text_lower",
            "text_no_punct",
            "text_tokenized",
            "text_no_stop",
            "text_lemmatized",
            "final",
        ]:
            samples = df_disaster_tweet[column].head(5).tolist()
            samples_formatted = "\n".join(
                f"{index}. {item}" for index, item in enumerate(samples, start=1)
            )
            logger.info(f"Sample from column '{column}':\n{samples_formatted}\n")

        data_disaster = df_disaster_tweet[df_disaster_tweet["target"] == 1]
        data_not_disaster = df_disaster_tweet[df_disaster_tweet["target"] == 0]

        logger.info("Data Disaster Shape: %s", data_disaster.shape)
        logger.info("Data Not Disaster Shape: %s", data_not_disaster.shape)
        logger.info("✅ Data preprocessing completed with success!")

        return {
            "df_disaster_tweet": df_disaster_tweet,
            "data_disaster": data_disaster,
            "data_not_disaster": data_not_disaster,
            "run_id": run_id 
        }
    except AttributeError as attr_error:
        logger.error(f"❌ AttributeError in text_preprocessing function: {attr_error}")
    except KeyError as key_error:
        logger.error(f"❌ KeyError in text_preprocessing function: {key_error}")
    except TypeError as type_error:
        logger.error(f"❌ TypeError in text_preprocessing function: {type_error}")
    except ValueError as value_error:
        logger.error(f"❌ ValueError in text_preprocessing function: {value_error}")
    except pd.errors.EmptyDataError as empty_data_error:
        logger.error(f"❌ EmptyDataError in text_preprocessing function: {empty_data_error}")
    except RuntimeError as runtime_error:
        logger.error(f"❌ RuntimeError in text_preprocessing function: {runtime_error}")
    except Exception as general_error:
        logger.error(f"❌ General error in text_preprocessing function: {general_error}")
    finally:
        run.finish()
    return {"df_disaster_tweet": None, "data_disaster": None, "data_not_disaster": None, "run_id": None}


def disaster_cloud(data_disaster, logger, run_id) -> None:
    """
    Generates and logs a WordCloud for disaster-related tweets.

    Args:
        data_disaster: DataFrame with disaster-related tweets.
        logger: Logger object.
        run: Wandb run object.
    """
    try:
        run = wandb.init(
            project="disaster_tweet_classification", id=run_id, resume=True
        )

        wordcloud_disaster = WordCloud(
            max_words=500, random_state=100, background_color="white", collocations=True
        ).generate(str((data_disaster["final"])))
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud_disaster, interpolation="bilinear")
        plt.title("WordCloud of the Disaster Tweets")
        plt.axis("off")
        disaster_path = "wordcloud_disaster.png"
        plt.savefig(disaster_path)
        plt.close()

        logger.info("⏳ Uploading Figure 1 - WordCloud Disaster Tweets")
        run.log({"WordCloud Disaster Tweets": wandb.Image(disaster_path)})

    except KeyError as key_error:
        logger.error(f"❌ KeyError in disaster_cloud function: {key_error}")
    except ValueError as value_error:
        logger.error(f"❌ ValueError in disaster_cloud function: {value_error}")
    except RuntimeError as runtime_error:
        logger.error(f"❌ RuntimeError in disaster_cloud function: {runtime_error}")
    except IOError as io_error:
        logger.error(f"❌ IOError in disaster_cloud function: {io_error}")
    except wandb.errors.CommError as comm_error:
        logger.error(f"❌ Wandb communication error in disaster_cloud function: {comm_error}")
    except Exception as general_error:
        logger.error(f"❌ General error in disaster_cloud function: {general_error}")
    finally:
        run.finish()


def non_disaster_cloud(data_not_disaster, logger, run_id) -> None:
    """
    Generates and logs a WordCloud for non-disaster-related tweets.

    Args:
        data_not_disaster: DataFrame with non-disaster-related tweets.
        logger: Logger object.
        run: Wandb run object.
    """
    try:
        run = wandb.init(
            project="disaster_tweet_classification", id=run_id, resume=True
        )
        if data_not_disaster is not None:
            wordcloud_not_disaster = WordCloud(
                max_words=500, random_state=100, background_color="white", collocations=True
            ).generate(str((data_not_disaster["final"])))
            plt.figure(figsize=(15, 10))
            plt.imshow(wordcloud_not_disaster, interpolation="bilinear")
            plt.title("WordCloud of the Non-Disaster Tweets")
            plt.axis("off")
            not_disaster_path = "wordcloud_not_disaster.png"
            plt.savefig(not_disaster_path)
            plt.close()

            logger.info("⏳ Uploading Figure 2 - WordCloud Non-Disaster Tweets")
            run.log({"WordCloud Non-Disaster Tweets": wandb.Image(not_disaster_path)})
    except IOError as io_error:
        logger.error(f"❌ IOError (saving image) in non_disaster_cloud function: {io_error}")
    except ValueError as value_error:
        logger.error(f"❌ ValueError in non_disaster_cloud function: {value_error}")
    except RuntimeError as runtime_error:
        logger.error(f"❌ RuntimeError in non_disaster_cloud function: {runtime_error}")
    except Exception as general_error:
        logger.error(f"❌ General error in non_disaster_cloud function: {general_error}")
    finally:
        run.finish()


def create_and_finalize_preprocessing_artifact(
    df_disaster_tweet, logger, run_id
) -> str or None:
    """
    Creates a Weights & Biases artifact for the preprocessed data, logs it, and
    finalizes the Wandb run.

    Args:
        df_disaster_tweet: DataFrame with the preprocessed data.
        logger: Logger object.
        run_id: Wandb run id.
    """
    try:
        # Initialize Wandb run
        run = wandb.init(
            project="disaster_tweet_classification", id=run_id, resume=True
        )

        # Create preprocessing artifact
        preprocessing_artifact = wandb.Artifact(
            "processed_data",
            type="Preprocessing",
            description="Preprocessing for Disaster-Related Tweets",
        )
        processed_data_path = "df_disaster_tweet_processed.csv"
        df_disaster_tweet.to_csv(processed_data_path, index=False)
        preprocessing_artifact.add_file(processed_data_path)

        # Log the artifact
        run.log_artifact(preprocessing_artifact)
        logger.info("✅ Preprocessing artifact logged successfully.")

        # Save the digest of the artifact for later use
        preprocessing_artifact_digest = preprocessing_artifact.digest

        logger.info("✅ Preprocessing artifact created with success!")
        return preprocessing_artifact_digest

    except Exception as error:
        logger.error(
            f"❌ Error during preprocessing artifact creation and logging: {error}"
        )
        return None
    finally:
        # Finish the run
        run.finish()
        logger.info("✅ Wandb run finished successfully.")
