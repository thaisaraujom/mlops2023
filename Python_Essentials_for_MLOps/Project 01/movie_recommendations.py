"""
Module for providing movie recommendations.

This module utilizes a content-based filtering approach to recommend movies based on 
their similarity to a given movie. It uses TF-IDF and cosine similarity to find matches 
amongst the dataset.
"""
import os
import re
import argparse
import logging
import zipfile
import requests
from termcolor import colored
from tabulate import tabulate
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data() -> None:
    """
    Download the MovieLens dataset.

    Returns:
        None
    """
    url = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
    zip_filename = "ml-25m.zip"
    extracted_folder_name = "ml-25m"

    if os.path.exists(extracted_folder_name):
        logging.info("📂 Data already exists. No need to download.")
        return

    try:
        # Download the file
        logging.info("🚀 Starting download from %s.", url)

        with requests.Session() as session:
            response = session.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            chunk_size = 128 * 1024
            total_chunks = total_size // chunk_size

            with open(zip_filename, 'wb') as file:
                for data in tqdm(response.iter_content(chunk_size=chunk_size),
                                total=total_chunks,
                                unit='KB',
                                unit_scale=True):
                    file.write(data)

        logging.info("✅ Download of %s completed.", zip_filename)

        # Unzip the file
        logging.info("🔓 Unzipping %s.", zip_filename)
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(".")
        logging.info("📂 Unzipping completed.")

        # Remove the ZIP file
        logging.info("🗑️ Removing %s.", zip_filename)
        os.remove(zip_filename)
        logging.info("🧹 %s removed.", zip_filename)

    except requests.ConnectionError:
        logging.error("❌ Failed to download due to a connection error. "
                      "Please check your internet connection. ❌")
    except requests.Timeout:
        logging.error("❌ Download timed out. Please try again. ❌")
    except requests.RequestException as error_request:
        logging.error("❌ An error occurred while downloading: %s ❌", error_request)
    except zipfile.BadZipFile:
        logging.error("❌ The downloaded file is not a valid ZIP file. ❌")

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError as error:
        logging.error('File not found: %s', error)
        return pd.DataFrame()

def clean_title(local_title: str) -> str:
    """
    Cleans the title of a movie by removing non-alphanumeric characters.
    
    Args:
        title (str): The original movie title.
        
    Returns:
        str: The cleaned movie title.
    """
    return re.sub("[^a-zA-Z0-9 ]", "", local_title)

def search(local_title: str) -> pd.DataFrame:
    """
    Searches for movies with titles similar to the given title.
    
    Args:
        title (str): The movie title to search for.
        
    Returns:
        pd.DataFrame: DataFrame with movies that have titles similar to the given title.
    """
    # Clean the given title using the previously defined function
    local_title = clean_title(local_title)
    # Transform the title into its TF-IDF representation
    query_vec = vectorizer.transform([local_title])
    # Compute the cosine similarity between the query and all movies
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    # Get indices of the top 5 most similar movies
    indices = np.argpartition(similarity, -5)[-5:]
    # Fetch the actual movie data for these indices
    results = movies.iloc[indices].iloc[::-1]

    return results

def filter_by_rating(dataframe: pd.DataFrame,
                                    rating_threshold: float = 4.0) -> pd.DataFrame:
    """
    Filters DataFrame based on a rating threshold.

    Args:
        dataframe (pd.DataFrame): DataFrame to filter.
        rating_threshold (float, optional): The rating threshold. Defaults to 4.0.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return dataframe[dataframe["rating"] > rating_threshold]

def find_similar_movies(local_movie_id: int) -> pd.DataFrame:
    """
    Finds movies similar to the given movie id.

    Args:
        local_movie_id (int): The ID of the movie to find similar movies for.

    Returns:
        pd.DataFrame: DataFrame with movies that are similar to the given movie.
    """

    local_similar_users = ratings[(ratings["movieId"] == local_movie_id)]
    local_similar_users = filter_by_rating(local_similar_users)["userId"].unique()

    local_similar_user_recs = ratings[ratings["userId"].isin(local_similar_users)]
    local_similar_user_recs = filter_by_rating(local_similar_user_recs)["movieId"]
    local_similar_user_recs = local_similar_user_recs.value_counts() / len(local_similar_users)

    local_similar_user_recs = local_similar_user_recs[local_similar_user_recs > .10]
    all_users = ratings[ratings["movieId"].isin(local_similar_user_recs.index)]
    all_users = filter_by_rating(all_users)
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    local_rec_percentages = pd.concat([local_similar_user_recs, all_user_recs], axis=1)
    local_rec_percentages.columns = ["similar", "all"]

    local_rec_percentages["score"] = local_rec_percentages["similar"] / local_rec_percentages["all"]
    local_rec_percentages = local_rec_percentages.sort_values("score", ascending=False)

    merged_data = local_rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")
    return merged_data[["score", "title", "genres"]]

def display_dataframe(_df: pd.DataFrame, _title: str = ""):
    """
    Nicely display a DataFrame using tabulate.
    
    Args:
        _df (pd.DataFrame): DataFrame to display.
        _title (str): Optional header title.
    """
    if _title:
        print(colored(_title, attrs=['bold']))

    print(tabulate(_df, headers='keys', tablefmt='grid', showindex=False))
    print()

def parse_arguments() -> argparse.Namespace:
    """"
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Movie Recommendations CLI")
    parser.add_argument('-t',
                        '--title',
                        help='Movie title to search or get recommendations for',
                        type=str,
                        required=True)
    return parser.parse_args()

if __name__ == "__main__":

    # Download the data
    download_data()

    # Load data and add exception handling
    movies = load_csv("./ml-25m/movies.csv")
    ratings = load_csv('./ml-25m/ratings.csv')

    # Display the first few rows of the movie dataset and ratings dataset
    display_dataframe(movies.head(), "First few rows of movies dataset")
    display_dataframe(ratings.head(), "First few rows of ratings dataset")

    # Apply the clean_title function to each movie title
    movies["clean_title"] = movies["title"].apply(clean_title)
    display_dataframe(movies.head(), "Movies dataset after title cleaning")

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(movies["clean_title"])
    args = parse_arguments()
    title = args.title

    # First, display search results based on title
    search_results = search(title)
    display_dataframe(search_results, "Search Results")

    # If we have search results, then show recommendations for the most relevant movie
    if not search_results.empty:
        movie_id = search_results.iloc[0]["movieId"]
        recommendations = find_similar_movies(movie_id)
        display_dataframe(recommendations, "Recommendations based on the most relevant movie")
