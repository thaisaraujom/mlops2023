"""
Predicting Heart Disease
"""

import time
import logging
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

BASE_PATH = '/workspaces/mlops2023/Python_Essentials_for_MLOps/Project 03/'
INFO_EMOJI = "ℹ️ "
ERROR_EMOJI = "❌ "
DEBUG_EMOJI = "🔍 "
SUCCESS_EMOJI = "✅ "
WARNING_EMOJI = "⚠️ "

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame.

    Args:
        file_path: Path to the CSV file.
    
    Returns:
        Pandas DataFrame containing the data from the CSV file.
    """
    try:
        logging.info('%s Loading data from file: %s', INFO_EMOJI, file_path)
        data = pd.read_csv(file_path)
        logging.info('%s Data loaded successfully. %s rows loaded.', SUCCESS_EMOJI, len(data))
        return data
    except FileNotFoundError as error_file:
        logging.error('%s File not found: %s', ERROR_EMOJI, error_file)
        raise
    finally:
        logging.info('%s Finished attempting to load data.', INFO_EMOJI)


def log_data_info(dataframe: pd.DataFrame) -> None:
    """
    Log some basic information about the data in a Pandas DataFrame.

    Args:
        dataframe: Pandas DataFrame containing the data.

    Returns:
        None
    """
    logging.info('%s Data types:\n%s', INFO_EMOJI, dataframe.dtypes)
    logging.info('%s Value counts of data types:\n%s', INFO_EMOJI, dataframe.dtypes.value_counts())
    logging.info('%s Descriptive statistics:\n%s', INFO_EMOJI, dataframe.describe())
    logging.info('%s Missing values count:\n%s', INFO_EMOJI, dataframe.isna().sum())
    logging.info(
        '%s Descriptive statistics for object data types:\n%s', 
        INFO_EMOJI, dataframe.describe(include=['object'])
    )


def plot_categorical_counts(dataframe: pd.DataFrame,
                            categorical_columns: list,
                            save_path: str) -> None:
    """
    Plot the counts of categorical columns in a Pandas DataFrame.

    Args:
        dataframe: Pandas DataFrame containing the data.
        categorical_columns: List of categorical columns in the DataFrame.
        save_path: Directory path where the plot will be saved.

    Returns:
        None
    """
    start_time = time.time()
    logging.info('%sPlotting categorical counts...', INFO_EMOJI)

    # Determine the layout dynamically based on the number of categorical columns
    columns = math.ceil(math.sqrt(len(categorical_columns)))
    rows = math.ceil(len(categorical_columns) / columns)

    plt.figure(figsize=(16, 15))
    for idx, col in enumerate(categorical_columns):
        if col not in dataframe.columns:
            logging.warning('%sColumn "%s" not found in dataframe.', WARNING_EMOJI, col)
            continue  # Skip this column if it's not found

        plot_axis = plt.subplot(rows, columns, idx + 1)
        sns.countplot(x=dataframe[col], ax=plot_axis)
        plot_axis.set_title(f'Count of {col}')
        # add data labels to each bar
        for container in plot_axis.containers:
            plot_axis.bar_label(container, label_type="center")

    plt.tight_layout()  # Adjusts the layout to prevent clipping
    plt.savefig(f'{save_path}categorical_plots.png')
    logging.info('%sPlots saved to %scategorical_plots.png', SUCCESS_EMOJI, save_path)
    logging.info('%s Plotting completed in %s seconds.', INFO_EMOJI, time.time() - start_time)


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the data in a Pandas DataFrame.

    Args:
        dataframe: Pandas DataFrame containing the data.

    Returns:
        Pandas DataFrame containing the cleaned data.
    """
    logging.debug('%s Starting data cleaning process.', DEBUG_EMOJI)
    df_clean = dataframe.copy()
    # only keep non-zero values for RestingBP
    df_clean = df_clean[df_clean["RestingBP"] != 0]
    heart_disease_mask = df_clean["HeartDisease"] == 0
    cholesterol_without_heart_disease = df_clean.loc[heart_disease_mask, "Cholesterol"]
    cholesterol_with_heart_disease = df_clean.loc[~heart_disease_mask, "Cholesterol"]
    df_clean.loc[heart_disease_mask, "Cholesterol"] = cholesterol_without_heart_disease.replace(
        to_replace=0, value=cholesterol_without_heart_disease.median())
    df_clean.loc[~heart_disease_mask, "Cholesterol"] = cholesterol_with_heart_disease.replace(
        to_replace=0, value=cholesterol_with_heart_disease.median())
    logging.debug('%s Data cleaning process completed.', DEBUG_EMOJI)
    return df_clean


def plot_correlation_heatmap(dataframe: pd.DataFrame) -> None:
    """
    Plot a correlation heatmap for the data in a Pandas DataFrame.

    Args:
        dataframe: Pandas DataFrame containing the data.

    Returns:
        None
    """
    correlations = abs(dataframe.corr())
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations, annot=True, cmap="Blues")
    plt.savefig(f'{BASE_PATH}correlation_heatmap.png')
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations[correlations > 0.3], annot=True, cmap="Blues")
    plt.savefig(f'{BASE_PATH}filtered_correlation_heatmap.png')


def train_knn(x_train: pd.DataFrame,
              x_val: pd.DataFrame,
              y_train: pd.Series,
              y_val: pd.Series,
              features: list) -> None:
    """
    Train a KNN classifier on the data in a Pandas DataFrame.

    Args:
        x_train: Pandas DataFrame containing the training data.
        x_val: Pandas DataFrame containing the validation data.
        y_train: Pandas Series containing the training labels.
        y_val: Pandas Series containing the validation labels.
        features: List of features to use for training.
    
    Returns:
        None
    """
    logging.info('%s Starting KNN training.', INFO_EMOJI)
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train[features])
    x_val_scaled = scaler.transform(x_val[features])
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train_scaled, y_train)
    accuracy = knn.score(x_val_scaled, y_val)
    logging.info('%s KNN training completed. Accuracy: %s', SUCCESS_EMOJI, accuracy * 100)


def grid_search_knn(x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """
    Grid search for the best KNN classifier.

    Args:
        x_train: Pandas DataFrame containing the training data.
        y_train: Pandas Series containing the training labels.
    
    Returns:
        GridSearchCV object containing the best KNN classifier.
    """
    logging.info('%s Starting grid search for KNN.', INFO_EMOJI)
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    grid_params = {"n_neighbors": range(1, 20),
                   "metric": ["minkowski", "manhattan"]
                   }
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, grid_params, scoring='accuracy')
    knn_grid.fit(x_train_scaled, y_train)
    logging.info('%s Grid search completed.', SUCCESS_EMOJI)
    return knn_grid


def evaluate_model(grid: GridSearchCV, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluate the best KNN classifier on the test set.

    Args:
        grid: GridSearchCV object containing the best KNN classifier.
        x_test: Pandas DataFrame containing the test data.
        y_test: Pandas Series containing the test labels.

    Returns:
        None
    """
    logging.info('%s Starting model evaluation.', INFO_EMOJI)
    scaler = MinMaxScaler()
    x_test_scaled = scaler.fit_transform(x_test)
    predictions = grid.best_estimator_.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    logging.info('%s Model evaluation completed. Accuracy: %s', SUCCESS_EMOJI, accuracy * 100)


def log_sex_distribution(data: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
    """
    Log the distributions of patients by sex.

    Args:
        data: Pandas DataFrame containing the entire dataset.
        x_train: Pandas DataFrame containing the training data.
        x_test: Pandas DataFrame containing the test data.
    
    Returns:
        None
    """
    logging.info("Distribution of patients by their sex in the entire dataset")
    logging.info('\n%s', data.Sex_M.value_counts())
    logging.info("\nDistribution of patients by their sex in the training dataset")
    logging.info('\n%s', x_train.Sex_M.value_counts())
    logging.info("\nDistribution of patients by their sex in the test dataset")
    logging.info('\n%s', x_test.Sex_M.value_counts())


def main():
    """
    Main function.
    """
    logging.info('%s Starting program execution.', INFO_EMOJI)
    file_path = "/workspaces/mlops2023/Python_Essentials_for_MLOps/Project 03/heart.csv"
    data = load_data(file_path)
    logging.info('%s First 5 rows of the dataset:\n%s',
                 INFO_EMOJI,
                 data.head())
    log_data_info(data)
    logging.info('%s Unique values in FastingBS and HeartDisease columns:\n%s %s',
                 INFO_EMOJI, data["FastingBS"].unique(),
                 data["HeartDisease"].unique())
    categorical_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG",
                        "ExerciseAngina", "ST_Slope", "HeartDisease"]
    plot_categorical_counts(data, categorical_cols, BASE_PATH)
    data_clean = clean_data(data)
    logging.info('%s Descriptive statistics for Cholesterol and RestingBP:\n%s',
                 INFO_EMOJI,
                 data_clean[["Cholesterol", "RestingBP"]].describe())
    data_clean = pd.get_dummies(data_clean, drop_first=True)
    logging.info('%s First 5 rows of the cleaned dataset:\n%s',
                 INFO_EMOJI,
                 data_clean.head())
    plot_correlation_heatmap(data_clean)

    feature_data = data_clean.drop(["HeartDisease"], axis=1)  # Renomeado x para feature_data
    label_data = data_clean["HeartDisease"]  # Renomeado y para label_data
    x_train, x_test, y_train, y_test = train_test_split(feature_data,
                                                        label_data,
                                                        test_size=0.15,
                                                        random_state=417)
    features = ["Oldpeak", "Sex_M", "ExerciseAngina_Y", "ST_Slope_Flat", "ST_Slope_Up"]

    # Train and evaluate k-NN with single features
    for feature in features:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train[[feature]], y_train)
        accuracy = knn.score(x_test[[feature]], y_test)
        logging.info(
            '%s The k-NN classifier trained on %s and with k = 3 '
            'has an accuracy of %.2f%%',
            INFO_EMOJI, feature, accuracy * 100)  # Updated this line
    # Train and evaluate k-NN with multiple features
    train_knn(x_train, x_test, y_train, y_test, features)

    # Grid search for optimal k-NN parameters
    grid = grid_search_knn(x_train[features], y_train)
    logging.info('%s Best score: %s, Best params: %s',
                 SUCCESS_EMOJI,
                 grid.best_score_,
                 grid.best_params_)

    # Evaluate the model with the test set
    evaluate_model(grid, x_test[features], y_test)

    # Print distribution of sex in datasets
    log_sex_distribution(feature_data, x_train, x_test)  # Atualizado x para feature_data
    logging.info('%s Program execution completed.', SUCCESS_EMOJI)

if __name__ == "__main__":
    main()
