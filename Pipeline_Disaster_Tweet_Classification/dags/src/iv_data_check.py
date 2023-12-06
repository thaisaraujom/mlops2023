"""
This file aims to check data related to 
disaster tweets before training
"""
import subprocess
import json
import pandas as pd
import wandb

def download_artifact(logger):
    """
    Download the processed_data artifact from Weights & Biases 
    and load it into a Pandas DataFrame
    """
    try:
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="data_check")
        local_path = run.use_artifact("processed_data:v0").download()
        df = pd.read_csv(f'{local_path}/df_disaster_tweet_processed.csv')
        logger.info('✅ Dataset for tests loaded with success!')

        return {'df': df, 'run_id': run.id}
    except Exception as e:
        logger.error(f'❌ An error occurred: {str(e)}')
        return {'df': None, 'run_id': None}

def test_columns_presence(data, logger, run_id):
    """
    Test if the required columns 'text' and 'target' are present in the DataFrame.
    """
    try:
        run = wandb.init(
            project="disaster_tweet_classification", id=run_id, resume=True
        )
        if 'text' in data.columns and 'target' in data.columns:
            logger.info('✅ Test 1 OK!')
            return True
        logger.error('❌ Failed Test 1')
        return False
    finally:
        run.finish()
        logger.info('✅ Test 1 finished!')


def test_columns_types(data, logger, run_id):
    """
    Test if the data types of 'text' and 'target' columns are as expected.
    """
    try:
        run = wandb.init(
            project="disaster_tweet_classification", id=run_id, resume=True
        )
        if data['text'].dtype == object and data['target'].dtype == int:
            logger.info('✅ Test 2 OK!')
            return True
        logger.error('❌ Failed Test 2')
        return False
    finally:
        run.finish()
        logger.info('✅ Test 2 finished!')


def test_data_length(data, logger, run_id):
    """
    Test if the length of the DataFrame is greater than 1000.
    """
    try:
        run = wandb.init(
            project="disaster_tweet_classification", id=run_id, resume=True
        )
        if len(data) > 1000:
            logger.info('✅ Test 3 OK!')
            return True
        logger.error('❌ Failed Test 3')
        return False
    finally:
        run.finish()
        logger.info('✅ Test 3 finished!')
        
def finalize_data_check(logger, run_id, test_1, test_2, test_3):
    """
    Finalize the data check process by creating an artifact, logging it, and finishing the Wandb run.

    Args:
        logger: Logger object.
        run_id: Wandb run id.
    """
    run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

    if run.resumed:
        logger.info("✅ Run returned successfully.")
    else:
        logger.warning("⚠️ Run does not exist. Starting a new one.")

    data_check_artifact = wandb.Artifact('data_check_final', type='Data Check',
                                    description='Final Data Check for Disaster-Related Tweets')
    if test_1 and test_2 and test_3:
            logger.info(f'{test_1+test_2+test_3}/3 tests passed!')
    else:
        logger.error(f'{test_1+test_2+test_3}/3 tests passed!')

    try:
        # Log the artifact
        run.log_artifact(data_check_artifact)
        logger.info("✅ Data Check artifact logged successfully.")

    except Exception as error_data_check:
        logger.error(f"❌ Error during the logging of Data Check artifact: {error_data_check}")
    
    finally:
        # Finish the run
        run.finish()
        logger.info("✅ Wandb run finished successfully.")