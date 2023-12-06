"""
This file aims to segregate data into
train, validation and test files
"""
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
import joblib

def download_artifact_preprocessed(logger):
    """
    Download the processed_data artifact from Weights & Biases 
    and load it into a Pandas DataFrame
    """
    try:
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="data_segregation")
        local_path = run.use_artifact("processed_data:v0").download()
        df = pd.read_csv(f'{local_path}/df_disaster_tweet_processed.csv')
        logger.info('✅ Dataset for data segregation loaded with success!')
        return {'df': df, 'run_id': run.id}
    except Exception as e:
        logger.error(f'❌ An error occurred: {str(e)}')
        return {'df': None, 'run_id': None}


def features_and_labels(df, logger, run_id):
    """
    Separate features (text) and labels (target)
    in the dataset
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)
        X = df['final'].tolist()
        y = df['target'].tolist()
        logger.info(f'Features: {X}')
        logger.info(f'Labels: {y}')
        return {'X': X, 'y': y, 'run_id': run.id}
    finally:
        run.finish()
        logger.info('✅ Features and labels segregation finished!')


def train_validation_test(X, y, logger, run_id):
    """
    Create train, validation and test datasets/arrays
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

        (train_x, test_x, train_y, test_y) = train_test_split(X, y,test_size=0.2, random_state=42)
        (train_x, val_x, train_y, val_y) = train_test_split(train_x, train_y,test_size=0.2, random_state=42)

        logger.info("Train x: {}".format(len(train_x)))
        logger.info("Train y: {}".format(len(train_y)))
        logger.info("Validation x: {}".format(len(val_x)))
        logger.info("Validation y: {}".format(len(val_y)))
        logger.info("Test x: {}".format(len(test_x)))
        logger.info("Test y: {}".format(len(test_y)))

        joblib.dump(train_x, 'train_x')
        joblib.dump(train_y, 'train_y')
        joblib.dump(val_x, 'val_x')
        joblib.dump(val_y, 'val_y')
        joblib.dump(test_x, 'test_x')
        joblib.dump(test_y, 'test_y')
        
        logger.info("Dumping the train and validation data artifacts to the disk")

        artifacts = {
            'train_x': ('train_data', 'A json file representing the train_x'),
            'train_y': ('train_data', 'A json file representing the train_y'),
            'val_x': ('val_data', 'A json file representing the val_x'),
            'val_y': ('val_data', 'A json file representing the val_y'),
            'test_x': ('test_data', 'A json file representing the test_x'),
            'test_y': ('test_data', 'A json file representing the test_y')
        }

        for artifact_name, (artifact_type, description) in artifacts.items():
            artifact = wandb.Artifact(artifact_name, type=artifact_type, description=description)
            logger.info(f"⏳ Logging {artifact_name} artifact")
            artifact.add_file(artifact_name)
            run.log_artifact(artifact)

        logger.info("✅ Artifacts logged successfully.")
    finally:
        run.finish()
        logger.info('✅ Train, validation and test data segregation finished!')
        logger.info("✅ Wandb run finished successfully.")