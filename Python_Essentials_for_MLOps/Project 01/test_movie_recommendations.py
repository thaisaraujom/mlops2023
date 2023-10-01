"""
Test the movie_recommendations.py file.
"""
import os
import pandas as pd
from movie_recommendations import clean_title

def test_clean_title() -> None:
    """
    Test the clean_title function.

    Returns:
        None
    """
    assert clean_title("Iron Man!") == "Iron Man"
    assert clean_title("Avengers: Endgame") == "Avengers Endgame"
    assert clean_title("Hello@#$%^&*()") == "Hello"

def test_file_format_and_existence() -> None:
    """
    Test the existence and format of the files.

    Returns:
        None
    """
    filepath = "./ml-25m/movies.csv"

    # Check if the file exists
    assert os.path.isfile(filepath), f"File {filepath} not found"

    # Check the file format
    assert filepath.endswith('.csv'), f"File {filepath} is not a CSV"

    # Check if the file isn't empty
    assert os.path.getsize(filepath) > 0, f"File {filepath} is empty"

def test_movies_csv_column_types() -> None:
    """
    Test the column types of the movies.csv file.

    Returns:
        None
    """
    movies = pd.read_csv("./ml-25m/movies.csv")

    # Check column data types
    assert movies["movieId"].dtype == "int64", "Incorrect type for column movieId"
    assert movies["title"].dtype == "object", "Incorrect type for column title"
    assert movies["genres"].dtype == "object", "Incorrect type for column genres"

def test_rating_csv_column_types():
    """
    Test the column types of the ratings.csv file.

    Returns:
        None
    """
    ratings = pd.read_csv("./ml-25m/ratings.csv")

    # Check column data types
    assert ratings["userId"].dtype == "int64", "Incorrect type for column userId"
    assert ratings["movieId"].dtype == "int64", "Incorrect type for column movieId"
    assert ratings["rating"].dtype == "float64", "Incorrect type for column rating"
    assert ratings["timestamp"].dtype == "int64", "Incorrect type for column timestamp"
