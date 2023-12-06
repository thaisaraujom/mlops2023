"""
This DAG is used to fetch and store the data from the disaster tweets project.
It is composed of two tasks: fetch_and_organize and store_fetch_data.
"""
from datetime import datetime, timedelta
from airflow.decorators import dag, task, task_group
from src.i_fetch_data import (
    setup_logging, fetch_and_organize_data, store_fetch_data
)
from src.ii_eda import (
    download_eda_artifact, general_info_dataset, create_bar_graph, 
    finalize_eda
)
from src.iii_preprocessing import (
    download_dataset_artifact, text_preprocessing, disaster_cloud, non_disaster_cloud,
    create_and_finalize_preprocessing_artifact
)
from src.iv_data_check import (
    download_artifact, test_columns_presence, test_columns_types,
    test_data_length, finalize_data_check
)
from src.v_data_segregation import (
    download_artifact_preprocessed, features_and_labels, train_validation_test
)
from src.vi_train import (
    download_artifacts, model1, model2, model3, model4,
    plot_loss_and_acc, 
)
from src.vii_test import (
    test_model1, test_model2, test_model3, test_model4, init_wandb
)

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 11, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag('disaster_tweets_pipeline', default_args=default_args, schedule_interval=None, catchup=False)
def disaster_tweets_pipeline():
    logger = setup_logging()

    @task_group(group_id='fetch_data_group')
    def fetch_data_tasks():
        @task
        def fetch_and_organize():
            return fetch_and_organize_data(logger)

        @task
        def store_data():
            return store_fetch_data(logger)

        fetch_task = fetch_and_organize()
        store_task = store_data()
        fetch_task >> store_task
        return store_task

    @task_group(group_id='eda_group')
    def eda_tasks():
        @task
        def download_eda_artifact_task():
            return download_eda_artifact(logger)

        @task
        def general_info_dataset_task(eda_artifact_result):
            general_info_dataset(logger, eda_artifact_result['df_disaster_tweet'], eda_artifact_result['run_id'])

        @task
        def create_bar_graph_task(eda_artifact_result):
            create_bar_graph(logger, eda_artifact_result['df_disaster_tweet'], eda_artifact_result['run_id'])

        @task
        def finalize_eda_task(eda_artifact_result):
            finalize_eda(logger, eda_artifact_result['run_id'])

        download_eda_artifact_result = download_eda_artifact_task()
        general_info_dataset_result = general_info_dataset_task(download_eda_artifact_result)
        create_bar_graph_result = create_bar_graph_task(download_eda_artifact_result)
        
        general_info_dataset_result >> create_bar_graph_result >> finalize_eda_task(download_eda_artifact_result)

    @task_group(group_id='preprocessing_group')
    def preprocessing_tasks():
        @task
        def download_dataset_artifact_task():
            return download_dataset_artifact(logger)

        @task
        def text_preprocessing_task(dataset_artifact_result):
            return text_preprocessing(dataset_artifact_result['df_disaster_tweet'], logger, dataset_artifact_result['run_id'])

        @task
        def disaster_cloud_task(preprocessing_result):
            return disaster_cloud(preprocessing_result['data_disaster'], logger, preprocessing_result['run_id'])

        @task
        def non_disaster_cloud_task(preprocessing_result):
            return non_disaster_cloud(preprocessing_result['data_not_disaster'], logger, preprocessing_result['run_id'])

        @task
        def create_and_finalize_preprocessing_artifact_task(preprocessing_result):
            return create_and_finalize_preprocessing_artifact(preprocessing_result['df_disaster_tweet'], logger, preprocessing_result['run_id'])

        download_dataset_artifact_result = download_dataset_artifact_task()
        preprocessing_result = text_preprocessing_task(download_dataset_artifact_result)
        disaster_cloud_result = disaster_cloud_task(preprocessing_result)
        non_disaster_cloud_result = non_disaster_cloud_task(preprocessing_result)
        create_and_finalize_preprocessing_artifact_result = create_and_finalize_preprocessing_artifact_task(preprocessing_result)

        download_dataset_artifact_result >> preprocessing_result
        preprocessing_result >> disaster_cloud_result >> non_disaster_cloud_result >> create_and_finalize_preprocessing_artifact_result

    @task_group(group_id='data_checks_group')
    def data_checks_tasks():
        @task
        def download_artifact_task():
            return download_artifact(logger)

        @task
        def test_columns_presence_task(artifact_results):
            return test_columns_presence(artifact_results['df'], logger, artifact_results['run_id'])

        @task
        def test_columns_types_task(artifact_results):
            return test_columns_types(artifact_results['df'], logger, artifact_results['run_id'])

        @task
        def test_data_length_task(artifact_results):
            return test_data_length(artifact_results['df'], logger, artifact_results['run_id'])

        @task
        def finalize_data_check_task(artifact_results, test1_result, test2_result, test3_result):
            return finalize_data_check(logger, artifact_results['run_id'], test1_result, test2_result, test3_result)

        artifact_results = download_artifact_task()
        test1_result = test_columns_presence_task(artifact_results)
        test2_result = test_columns_types_task(artifact_results)
        test3_result = test_data_length_task(artifact_results)
        finalize_data_check_task(artifact_results, test1_result, test2_result, test3_result)

    @task_group(group_id='data_segregation_group')
    def data_segregation_tasks():
        @task
        def download_artifact_task():
            return download_artifact_preprocessed(logger)

        @task
        def features_and_labels_task(artifact_result):
            return features_and_labels(artifact_result['df'], logger, artifact_result['run_id'])

        @task
        def train_validation_test_task(features_labels_result):
            return train_validation_test(features_labels_result['X'], features_labels_result['y'], logger, features_labels_result['run_id'])

        artifact_result = download_artifact_task()
        features_labels_result = features_and_labels_task(artifact_result)
        segregation_result = train_validation_test_task(features_labels_result)

        artifact_result >> features_labels_result >> segregation_result

    @task_group(group_id='training_group')
    def training_tasks():
        @task
        def download_artifacts_task():
            return download_artifacts(logger)

        @task
        def model1_task(artifact_result):
            return model1(artifact_result['train_x'], artifact_result['train_y'], 
                          artifact_result['val_x'], artifact_result['val_y'], 
                          artifact_result['run_id'], logger)

        @task
        def model2_task(artifact_result):
            return model2(artifact_result['train_x'], artifact_result['train_y'], 
                          artifact_result['val_x'], artifact_result['val_y'], 
                          artifact_result['run_id'], logger)

        @task
        def model3_task(artifact_result):
            return model3(artifact_result['train_x'], artifact_result['train_y'], 
                          artifact_result['val_x'], artifact_result['val_y'], 
                          artifact_result['run_id'], logger)

        @task
        def model4_task(artifact_result):
            return model4(artifact_result['train_x'], artifact_result['train_y'], 
                          artifact_result['val_x'], artifact_result['val_y'], 
                          artifact_result['run_id'], logger)

        artifact_results = download_artifacts_task()
        model1_results = model1_task(artifact_results)
        model2_results = model2_task(artifact_results)
        model3_results = model3_task(artifact_results)
        model4_results = model4_task(artifact_results)

        artifact_results >> model1_results >> model2_results >> model3_results >> model4_results

    @task_group(group_id='test_group')
    def test_tasks():
        @task
        def init_wandb_task():
            return init_wandb(logger)

        @task
        def test_model1_task(init_wandb_result):
            run_id = init_wandb_result['run_id']
            test_x = init_wandb_result['test_x']
            test_y = init_wandb_result['test_y']
            return test_model1(run_id, logger, test_x, test_y)

        @task
        def test_model2_task(init_wandb_result):
            run_id = init_wandb_result['run_id']
            test_x = init_wandb_result['test_x']
            test_y = init_wandb_result['test_y']
            return test_model2(run_id, logger, test_x, test_y)
        
        @task
        def test_model3_task(init_wandb_result):
            run_id = init_wandb_result['run_id']
            test_x = init_wandb_result['test_x']
            test_y = init_wandb_result['test_y']
            return test_model3(run_id, logger, test_x, test_y)
        
        @task
        def test_model4_task(init_wandb_result):
            run_id = init_wandb_result['run_id']
            test_x = init_wandb_result['test_x']
            test_y = init_wandb_result['test_y']
            return test_model4(run_id, logger, test_x, test_y)

        wandb_run_id = init_wandb_task()
        test_model1_task(wandb_run_id)
        test_model2_task(wandb_run_id)
        test_model3_task(wandb_run_id)
        test_model4_task(wandb_run_id)

    fetch_data_group_result = fetch_data_tasks()
    eda_group_result = eda_tasks()
    preprocessing_group_result = preprocessing_tasks()
    data_checks_group_result = data_checks_tasks()
    data_segregation_group_result = data_segregation_tasks()
    train_group_result = training_tasks()
    test_group_result = test_tasks()

    fetch_data_group_result >> eda_group_result >> preprocessing_group_result >> data_checks_group_result >> data_segregation_group_result >> train_group_result >> test_group_result

DAG = disaster_tweets_pipeline()