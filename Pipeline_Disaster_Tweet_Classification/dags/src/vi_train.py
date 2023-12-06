"""
This file aims to train disaster
tweets dataset with 4 different
models
"""
import wandb
import joblib
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import json
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L1, L2, L1L2
from tensorflow.keras.optimizers import RMSprop


def download_artifacts(logger):
    """
    Download train_x, train_y, val_x, val_y
    and vocab artifacts from Wandb
    """
    try:
        # Start run in wandb
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="train")

        # Download train_y
        train_x_artifact = run.use_artifact('train_x:latest')
        train_x_path = train_x_artifact.file()
        train_x = joblib.load(train_x_path)

        # Download train_y
        train_y_artifact = run.use_artifact('train_y:latest')
        train_y_path = train_y_artifact.file()
        train_y = joblib.load(train_y_path)

        # Download val_x
        val_x_artifact = run.use_artifact('val_x:latest')
        val_x_path = val_x_artifact.file()
        val_x = joblib.load(val_x_path)

        # Download val_y
        val_y_artifact = run.use_artifact('val_y:latest')
        val_y_path = val_y_artifact.file()
        val_y = joblib.load(val_y_path)

        # Verify the shape of the data
        logger.info(f'Train X shape: {len(train_x)}')
        logger.info(f'Train y shape: {len(train_y)}')
        logger.info(f'Val X shape: {len(val_x)}')
        logger.info(f'Val y shape: {len(val_y)}')

        # Verify the data
        logger.info(f'Train X: {train_x}')
        logger.info(f'Train y: {train_y}')
        logger.info(f'Val X: {val_x}')
        logger.info(f'Val y: {val_y}')

        # save val x to json
        with open('val_x.json', 'w') as f:
            json.dump(val_x, f)

        if all(isinstance(x, str) for x in train_x):
            logger.info('✅ Train X is a list of strings')
        else:
            logger.info('❌ Train X is not a list of strings')

        if all(isinstance(x, str) for x in val_x):
            logger.info('✅ Val x is a list of strings')
        else:
            logger.info('❌ Val x is not a list of strings')
        
        if all(item in [0, 1] for item in train_y):
            logger.info('✅ Train y has only 0 and 1 values')
        else:
            logger.info('❌ Train y has values different from 0 and 1')

        if all(item in [0, 1] for item in val_y):
            logger.info('✅ Val y has only 0 and 1 values')
        else:
            logger.info('❌ Val y has values different from 0 and 1')

        logger.info('✅ Artifacts downloaded with success!')
        return {'run_id': run.id, 'train_x': train_x, 'train_y': train_y, 'val_x': val_x, 'val_y': val_y}
    except Exception as e:
        logger.error(f'❌ An error occurred: {str(e)}')
        return {'run_id': None, 'train_x': None, 'train_y': None, 'val_x': None, 'val_y': None}


def plot_loss_and_acc(history, epochs, name_model, run, logger):
    """
    Plot loss and accuracy for each epoch
    while training the neural network
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1,1,figsize=(10,8))

    ax.plot(np.arange(0, epochs), history.history["loss"], label="train_loss",linestyle='--')
    ax.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss",linestyle='--')
    ax.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    ax.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    ax.set_title(f"Training Loss and Accuracy {name_model}")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend()
    plt.show()

    image_path = f'{name_model}_train_graph.png'
    fig.savefig(image_path)
    plt.close()
    run.log({f"Loss and Acc {name_model}": wandb.Image(image_path)})
    logger.info('✅ Plot loss and accuracy finished!')


def model1(X_train, y_train, x_val, y_val, run_id, logger):
    """
    Define the first model, a shallow neural
    network
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)
        max_tokens = 7500
        input_length = 128
        output_dim = 128
        vectorizer_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                            output_mode='int',
                                                            standardize='lower_and_strip_punctuation',
                                                            output_sequence_length=input_length)
        vectorizer_layer.adapt(X_train)    

        embedding_layer = Embedding(input_dim=max_tokens, 
                                    output_dim=output_dim, 
                                    input_length=input_length)
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
        model.add(vectorizer_layer)
        model.add(embedding_layer)
        model.add(tf.keras.layers.GlobalMaxPooling1D())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        opt = tf.optimizers.Adam(learning_rate=0.01)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        logger.info('⏳ Starting model 1 training')

        history = model.fit(X_train, y_train, epochs=10, verbose=2, validation_data=(x_val,y_val),
                callbacks=[wandb.keras.WandbCallback(save_model=False, compute_flops=True)])
        
        logger.info('✅ Model 1 training finished with success!')
        plot_loss_and_acc(history, 10, "Model1", run, logger)

        model_artifact = wandb.Artifact('model1', type='model')
        model.save(wandb.run.dir + '/model_tf', save_format='tf')
        model_artifact.add_dir(wandb.run.dir + '/model_tf')
        run.log_artifact(model_artifact)
        logger.info('✅ Model 1 saved with success!')

    finally:
        run.finish()
        logger.info('✅ Model 1 finished!')


def model2(X_train, y_train, x_val, y_val, run_id, logger):
    """
    Define the second model, a multi-layer
    neural network
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)
        
        max_tokens = 7500
        input_length = 128
        output_dim = 128
        vectorizer_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                            output_mode='int',
                                                            standardize='lower_and_strip_punctuation',
                                                            output_sequence_length=input_length)
        vectorizer_layer.adapt(X_train)
        embedding_layer = Embedding(input_dim=max_tokens, 
                                    output_dim=output_dim, 
                                    input_length=input_length)

        model_regularized = tf.keras.models.Sequential()
        model_regularized.add(tf.keras.Input(shape=(1,), dtype=tf.string))
        model_regularized.add(vectorizer_layer)
        model_regularized.add(embedding_layer)
        model_regularized.add(tf.keras.layers.GlobalAveragePooling1D())
        model_regularized.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=L1(0.0005)))
        model_regularized.add(tf.keras.layers.Dropout(0.6))
        model_regularized.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=L1L2(0.0005)))
        model_regularized.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=L2(0.0005)))
        model_regularized.add(tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=L2(0.0005)))
        model_regularized.add(tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=L2(0.0005)))
        model_regularized.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model_regularized.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        logger.info('⏳ Starting model 2 training')

        history = model_regularized.fit(X_train, y_train, epochs=10, verbose=2, validation_data=(x_val,y_val),
                callbacks=[wandb.keras.WandbCallback(save_model=False, compute_flops=True)])
        
        logger.info('✅ Model 2 training finished with success!')
        plot_loss_and_acc(history, 10, "Model2", run, logger)

        model_artifact = wandb.Artifact('model2', type='model')
        model_regularized.save(wandb.run.dir + '/model_tf', save_format='tf')
        model_artifact.add_dir(wandb.run.dir + '/model_tf')
        run.log_artifact(model_artifact)
        logger.info('✅ Model 2 saved with success!')

    finally:
        run.finish()
        logger.info('✅ Model 2 finished!')        


def model3(X_train, y_train, x_val, y_val, run_id, logger):
    """
    Define the third model, a Multilayer Bidirectional
    LSTM neural network
    """
    max_tokens = 7500
    input_length = 128
    output_dim = 128
    vectorizer_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                        output_mode='int',
                                                        standardize='lower_and_strip_punctuation',
                                                        output_sequence_length=input_length)
    vectorizer_layer.adapt(X_train)
    embedding_layer = Embedding(input_dim=max_tokens, 
                                output_dim=output_dim, 
                                input_length=input_length)
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)
        ml_bi_lstm = Sequential()
        ml_bi_lstm.add(Input(shape=(1,), dtype=tf.string))
        ml_bi_lstm.add(vectorizer_layer)
        ml_bi_lstm.add(embedding_layer)
        ml_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True)))
        ml_bi_lstm.add(Bidirectional(LSTM(128, return_sequences=True)))
        ml_bi_lstm.add(Bidirectional(LSTM(64)))
        ml_bi_lstm.add(Dense(64, activation='elu', kernel_regularizer=L1L2(0.0001)))
        ml_bi_lstm.add(Dense(32, activation='elu', kernel_regularizer=L2(0.0001)))
        ml_bi_lstm.add(Dense(8, activation='elu', kernel_regularizer=L2(0.0005)))
        ml_bi_lstm.add(Dense(8, activation='elu'))
        ml_bi_lstm.add(Dense(4, activation='elu'))
        ml_bi_lstm.add(Dense(1, activation='sigmoid'))

        opt = RMSprop(learning_rate=0.0001, rho=0.8, momentum=0.9)
        ml_bi_lstm.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        logger.info('⏳ Starting model 3 training')

        history = ml_bi_lstm.fit(X_train, y_train, epochs=3, validation_data=(x_val,y_val),
            callbacks=[wandb.keras.WandbCallback(save_model=False, compute_flops=True)])
        
        logger.info('✅ Model 3 training finished with success!')
        plot_loss_and_acc(history, 3, "Model3", run, logger)

        model_artifact = wandb.Artifact('model3', type='model')
        ml_bi_lstm.save(wandb.run.dir + '/model_tf', save_format='tf')
        model_artifact.add_dir(wandb.run.dir + '/model_tf')
        run.log_artifact(model_artifact)
        logger.info('✅ Model 3 saved with success!')
    
    finally:
        run.finish()
        logger.info('✅ Model 3 finished!')

def model4(train_x, train_y, val_x, val_y, run_id, logger):
    """
    Define the fourth model, a Transformer
    """
    max_tokens = 7500
    input_length = 128
    output_dim = 128
    vectorizer_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                        output_mode='int',
                                                        standardize='lower_and_strip_punctuation',
                                                        output_sequence_length=input_length)
    vectorizer_layer.adapt(train_x)

    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

        # Tokenize the text data
        train_encodings = tokenizer(list(train_x), truncation=True, padding=True)
        val_encodings = tokenizer(list(val_x), truncation=True, padding=True)

        # Convert the labels to numpy arrays
        train_labels = np.array(train_y)
        val_labels = np.array(val_y)

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        ))

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            val_labels
        ))

        train_dataset = train_dataset.batch(16)
        val_dataset = val_dataset.batch(16)

        model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)

        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        logger.info('⏳ Starting model 4 training')


        history = model.fit(train_dataset, epochs=2, validation_data=val_dataset)

        logger.info('✅ Model 4 training finished with success!')
        plot_loss_and_acc(history, 2, "Model4", run, logger)

        model_artifact = wandb.Artifact('model4', type='model')
        model.save(wandb.run.dir + '/model_tf', save_format='tf')
        model_artifact.add_dir(wandb.run.dir + '/model_tf')
        run.log_artifact(model_artifact)
        logger.info('✅ Model 4 saved with success!')
    
    finally:
        run.finish()
        logger.info('✅ Model 4 finished!')