"""
This file conducts an Exploratory Data Analysis 
(EDA) on disaster-related tweets.
"""
import pandas as pd
import wandb

def download_eda_artifact(logger) -> None:
    """
    Initialize a wandb run, download an artifact, and load a dataset for EDA.

    Args:
        logger: Logger object.
    """
    try:
        # Initialize wandb run
        run = wandb.init(project='disaster_tweet_classification', save_code=True, job_type="eda")

        # Get the artifact
        artifact = run.use_artifact('dataset:v0')

        # Download the content of the artifact to the local directory
        artifact_dir = artifact.download()

        # Path to dataset
        data_path = artifact_dir + '/train.csv'

        # Load dataset
        df_disaster_tweet = pd.read_csv(data_path)

        return {'df_disaster_tweet': df_disaster_tweet, 'run_id': run.id}
    except FileNotFoundError as fnf_error:
        logger.error(f"❌ File not found error during artifact download: {fnf_error}")
    except pd.errors.ParserError as parse_error:
        logger.error(f"❌ Parsing error during data loading: {parse_error}")
    except wandb.errors.CommError as wandb_error:
        logger.error(f"❌ Wandb communication error: {wandb_error}")
    except Exception as error_eda:
        logger.error(f"❌ Unexpected error during artifact download: {error_eda}")
    return {'df_disaster_tweet': None, 'run_id': None}

def general_info_dataset(logger, df_disaster_tweet, run_id) -> wandb.Artifact:
    """
    Display general information about the dataset to wandb.

    Args:
        logger: Logger object.
        df_disaster_tweet: DataFrame with the dataset.
        run_id: Wandb run id.
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

        if run.resumed:
            logger.info("✅ Run returned successfully.")
        else:
            logger.warning("⚠️ Run does not exist. Starting a new one.")

        # Log the shape of the DataFrame
        run.log({'DataFrame Shape': df_disaster_tweet.shape})
        logger.info('DataFrame Shape: %s', df_disaster_tweet.shape)

        # Convert the DataFrame to a dictionary and log it
        run.log({"Head": df_disaster_tweet.head().to_dict('records')})
        run.log({"Tail": df_disaster_tweet.tail().to_dict('records')})

        # Create tables in Wandb with the columns of the dataset, its head and its tail
        columns_df = pd.DataFrame({'Column': df_disaster_tweet.columns})
        logger.info('Columns: %s', columns_df)
        run.log({'Columns Table': wandb.Table(dataframe=columns_df)})
        logger.info('Head: %s', df_disaster_tweet.head())
        run.log({'Head Table': wandb.Table(dataframe=df_disaster_tweet.head())})
        logger.info('Tail: %s', df_disaster_tweet.tail())
        run.log({'Tail Table': wandb.Table(dataframe=df_disaster_tweet.tail())})

        # Log unique value counts
        unique_counts = {
            'Keyword Unique': df_disaster_tweet['keyword'].nunique(),
            'Location Unique': df_disaster_tweet['location'].nunique(),
            'Text Unique': df_disaster_tweet['text'].nunique(),
            'ID Unique': df_disaster_tweet['id'].nunique(),
            'Target Unique': df_disaster_tweet['target'].nunique()
        }

        # v is the number of unique values for each column
        # k is the name of the column
        for k, v in unique_counts.items():
            run.log({k: v})
            logger.info('%s: %s', k, v)
    except pd.errors.EmptyDataError as empty_data_error:
        logger.error(f"❌ Empty data error: {empty_data_error}")
    except ValueError as value_error:
        logger.error(f"❌ Value error: {value_error}")
    except Exception as error_info:
        logger.error(f"❌ Unexpected error in general info dataset: {error_info}")
    finally:
        run.finish()


def create_bar_graph(logger, df_disaster_tweet, run_id) -> None:
    """
    Create a bar graph and log it to wandb.

    Args:
        logger: Logger object.
        df_disaster_tweet: DataFrame with the dataset.
        run_id: Wandb run id.
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

        if run.resumed:
            logger.info("✅ Run returned successfully.")
        else:
            logger.warning("⚠️ Run does not exist. Starting a new one.")
        # Drop unnecessary columns
        df_disaster_tweet = df_disaster_tweet.drop(['id', 'keyword', 'location'], axis=1)

        # Log target counts and proportions
        target_counts = df_disaster_tweet['target'].value_counts().reset_index()
        target_proportion = df_disaster_tweet['target'].value_counts(normalize=True).reset_index()

        # Converting the result to a dictionary and logging it
        run.log({"Target Counts": target_counts.to_dict('records')})
        run.log({"Target Proportion": target_proportion.to_dict('records')})

        # Create tables in Wandb with Target Counts and Target Proportion
        run.log({'Target Counts Table': wandb.Table(dataframe=target_counts)})
        run.log({'Target Proportion Table': wandb.Table(dataframe=target_proportion)})

        # Convert the data to a format compatible with wandb.Table
        data = [[label, count] for label, count in
                df_disaster_tweet['target'].value_counts().items()]
        table = wandb.Table(data=data, columns=["label", "count"])

        # Use the wandb plotting function to create the count plot
        wandb.log({"Target Count Plot":
                    wandb.plot.bar(table, "label", "count", title="Distribution of Target")})

    except pd.errors.EmptyDataError as empty_data_error:
        logger.error(f"❌ Empty data error: {empty_data_error}")
    except ValueError as value_error:
        logger.error(f"❌ Value error: {value_error}")
    except Exception as error_bar:
        logger.error(f"❌ Unexpected error creating bar graph: {error_bar}")
    finally:
        run.finish()

def finalize_eda(logger, run_id) -> None:
    """
    Finalize the EDA process by creating an artifact, logging it, and finishing the Wandb run.

    Args:
        logger: Logger object.
        run_id: Wandb run id.
    """
    run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

    if run.resumed:
        logger.info("✅ Run returned successfully.")
    else:
        logger.warning("⚠️ Run does not exist. Starting a new one.")

    eda_artifact = wandb.Artifact('eda_final', type='EDA',
                                    description='Final EDA for Disaster-Related Tweets')

    try:
        # Log the artifact
        run.log_artifact(eda_artifact)
        logger.info("✅ EDA artifact logged successfully.")

    except Exception as error_eda:
        logger.error(f"❌ Error during the logging of EDA artifact: {error_eda}")
    
    finally:
        # Finish the run
        run.finish()
        logger.info("✅ Wandb run finished successfully.")
