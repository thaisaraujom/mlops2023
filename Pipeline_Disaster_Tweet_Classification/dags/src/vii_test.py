import wandb
import joblib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoTokenizer


def init_wandb(logger):
    """
    Initialize wandb
    """
    try:
        run = wandb.init(project='disaster_tweet_classification',
                         save_code=True, job_type="test")

        # Download test_x
        test_x_artifact = run.use_artifact('test_x:latest')
        test_x_path = test_x_artifact.file()
        test_x = joblib.load(test_x_path)

        # Download test_y
        test_y_artifact = run.use_artifact('test_y:latest')
        test_y_path = test_y_artifact.file()
        test_y = joblib.load(test_y_path)

        logger.info("✅ Wandb initialization finished!")
        return {'run_id': run.id, 'test_x': test_x, 'test_y': test_y}
    except Exception as e:
        logger.error("Wandb initialization failed!")
        logger.error(e)
        raise e
        return {'run_id': None, 'test_x': None, 'test_y': None}

def test_model1(run_id, logger, test_x, test_y):
    """
    Tests for model 1
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

        # Download model 1
        model1 = run.use_artifact('model1:latest').download()
        model1 = tf.keras.models.load_model(model1)

        # Model evaluation
        metrics = model1.evaluate(test_x, test_y)
        logger.info(f'Model 1 - Test loss: {round(metrics[0], 3)}')
        logger.info(f'Model 1 - Accuracy: {round(metrics[1], 3)}')

    finally:
        run.finish()
        logger.info('✅ Model 1 test finished!')

def test_model2(run_id, logger, test_x, test_y):
    """
    Tests for model 2
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

        # Download model 2
        model2 = run.use_artifact('model2:latest').download()
        model2 = tf.keras.models.load_model(model2)

        # Model evaluation
        metrics = model2.evaluate(test_x, test_y)
        logger.info(f'Model 2 - Test loss: {round(metrics[0], 3)}')
        logger.info(f'Model 2 - Accuracy: {round(metrics[1], 3)}')

        # Predictions
        predictions = model2.predict(test_x)
        predictions = (predictions > 0.5).astype(int)

        # Classification report
        logger.info(classification_report(test_y,predictions))

        # Confusion matrix
        fig, ax = plt.subplots(1,1,figsize=(7,4))
        ConfusionMatrixDisplay(confusion_matrix(predictions, test_y)).plot(values_format=".0f",ax=ax)
        ax.set_xlabel("True Label")
        ax.set_ylabel("Predicted Label")
        ax.grid(False)
        plt.show()

        image_path = 'model2_confusion_matrix.png'
        fig.savefig(image_path)
        plt.close()
        run.log({"Confusion Matrix Model 2": wandb.Image(image_path)})
    
    finally:
        run.finish()
        logger.info('✅ Model 2 test finished!')


def test_model3(run_id, logger, test_x, test_y):
    """
    Tests for model 3
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

        # Download model 2
        model3 = run.use_artifact('model3:latest').download()
        model3 = tf.keras.models.load_model(model3)

        # Model evaluation
        metrics = model3.evaluate(test_x, test_y)
        logger.info(f'Model 3 - Test loss: {round(metrics[0], 3)}')
        logger.info(f'Model 3 - Accuracy: {round(metrics[1], 3)}')

        # Predictions
        predictions = model3.predict(test_x)
        predictions = (predictions > 0.5).astype(int)

        # Classification report
        logger.info(classification_report(test_y,predictions))

        # Confusion matrix
        fig, ax = plt.subplots(1,1,figsize=(7,4))
        ConfusionMatrixDisplay(confusion_matrix(predictions, test_y)).plot(values_format=".0f",ax=ax)
        ax.set_xlabel("True Label")
        ax.set_ylabel("Predicted Label")
        ax.grid(False)
        plt.show()

        image_path = 'model3_confusion_matrix.png'
        fig.savefig(image_path)
        plt.close()
        run.log({"Confusion Matrix Model 3": wandb.Image(image_path)})

    finally:
        run.finish()
        logger.info('✅ Model 3 test finished!')


def test_model4(run_id, logger, test_x, test_y):
    """
    Tests for model 4
    """
    try:
        run = wandb.init(project='disaster_tweet_classification', id=run_id, resume=True)

        # Download model 4
        model4 = run.use_artifact('model4:latest').download()
        model4 = tf.keras.models.load_model(model4)

        # logger prediction
        logger.info("Model 4 - Predictions started...")
        # Predictions
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        tokenized = tokenizer(list(test_x), truncation=True, padding=True)
        outputs = model4(tokenized)
        classifications = np.argmax(outputs['logits'], axis=1)
        logger.info("Model 4 - Predictions finished!")

        # Classification report
        logger.info(classification_report(test_y, classifications))

        # Confusion matrix
        fig, ax = plt.subplots(1,1,figsize=(7,4))
        ConfusionMatrixDisplay(confusion_matrix(classifications, test_y)).plot(values_format=".0f",ax=ax)
        ax.set_xlabel("True Label")
        ax.set_ylabel("Predicted Label")
        ax.grid(False)
        plt.show()

        image_path = 'model4_confusion_matrix.png'
        fig.savefig(image_path)
        plt.close()
        run.log({"Confusion Matrix Model 4": wandb.Image(image_path)})

    finally:
        run.finish()
        logger.info('✅ Model 4 test finished!')