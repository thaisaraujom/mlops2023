"""
Test the heart_disease_prediction.py file.
"""
import pandas as pd
from heart_disease_prediction import load_data, clean_data

# Load data from CSV file
DATA = load_data('/workspaces/mlops2023/Python_Essentials_for_MLOps/Project 03/heart.csv')

def test_load_data():
    """
    Test if data is loaded correctly.
    
    Returns:
        None
    """
    assert isinstance(DATA, pd.DataFrame)
    assert not DATA.empty

def test_clean_data():
    """
    Clean the loaded data and verify the cleaning process.
    
    Returns:
        None
    """
    cleaned_data = clean_data(DATA)
    assert isinstance(cleaned_data, pd.DataFrame)
    assert not cleaned_data.empty
    assert all(cleaned_data['RestingBP'] != 0)

def test_dataframe_properties():
    """
    Load data and check for certain DataFrame properties like expected columns and data types.
    
    Returns:
        None
    """
    # Check if the expected columns are present
    expected_columns = [
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
        'ST_Slope', 'HeartDisease'
    ]
    actual_columns = list(DATA.columns)
    error_message = (
        f"Expected columns: {expected_columns}, but got: {actual_columns}"
    )
    assert actual_columns == expected_columns, error_message

    # Check the data types of columns
    expected_dtypes = {
        'Age': 'int64',
        'Sex': 'object',
        'ChestPainType': 'object',
        'RestingBP': 'int64',
        'Cholesterol': 'int64',
        'FastingBS': 'int64',
        'RestingECG': 'object',
        'MaxHR': 'int64',
        'ExerciseAngina': 'object',
        'Oldpeak': 'float64',
        'ST_Slope': 'object',
        'HeartDisease': 'int64'
    }
    actual_dtypes = DATA.dtypes.to_dict()
    error_message = (
        f"Expected dtypes: {expected_dtypes}, but got: {actual_dtypes}"
    )
    assert actual_dtypes == expected_dtypes, error_message
    # Check for null values
    assert not DATA.isnull().any().any(), "There are null values in the DataFrame"
