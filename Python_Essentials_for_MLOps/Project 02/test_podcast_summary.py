"""
Test the podcast_summary.py file
"""
import os
from unittest.mock import patch, MagicMock
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
import pandas as pd
import requests
from podcast_summary import (
    create_database,
    fetch_episodes,
    load_episodes_logic,
    download_episodes_logic
)

# Sample data for testing
EPISODES_DATA = [
    {
        "link": "https://example.com/episode1",
        "title": "Episode 1",
        "pubDate": "2023-09-30",
        "description": "Description 1",
        "enclosure": {"@url": "https://example.com/episode1.mp3"}
    },
    {
        "link": "https://example.com/episode2",
        "title": "Episode 2",
        "pubDate": "2023-09-29",
        "description": "Description 2",
        "enclosure": {"@url": "https://example.com/episode2.mp3"}
    }
]

def test_create_database() -> None:
    """
    Test that create_database returns an instance of SQLExecuteQueryOperator.
    """
    result = create_database()
    assert isinstance(result, SQLExecuteQueryOperator)

def test_fetch_episodes() -> None:
    """
    Test that fetch_episodes returns a list of episodes.
    """
    with patch('podcast_summary.requests.get') as mock_get, \
         patch('podcast_summary.xmltodict.parse') as mock_parse:

        # Mock the response from requests.get
        mock_get.return_value.text = '<xml>fake xml</xml>'

        # Mock the result from xmltodict.parse
        mock_parse.return_value = {
            "rss": {
                "channel": {
                    "item": [
                        {"title": "Episode 1"},
                        {"title": "Episode 2"},
                    ]
                }
            }
        }

        # Call fetch_episodes
        result = fetch_episodes()

        # Check the result
        assert len(result) == 2
        assert result[0]['title'] == 'Episode 1'
        assert result[1]['title'] == 'Episode 2'

def mock_get_pandas_df(*_args, **_kwargs) -> pd.DataFrame:
    """
    Mocking the get_pandas_df method of SqliteHook.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    
    Returns:
        pd.DataFrame: A dataframe with the first episode.
    """
    # Simulating a database with only the first episode
    return pd.DataFrame([EPISODES_DATA[0]])

def mock_insert_rows(*_args, **_kwargs) -> None:
    """
    Mocking the insert_rows method of SqliteHook.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    
    Returns:
        None
    """
    pass  # Just a placeholder, as we're not actually inserting rows in this test

def test_load_episodes_logic(monkeypatch: MagicMock) -> None:
    """
    Test that load_episodes_logic returns the correct number of new episodes.
    """
    # Patching the methods of SqliteHook to use our mock functions
    monkeypatch.setattr(SqliteHook, "get_pandas_df", mock_get_pandas_df)
    monkeypatch.setattr(SqliteHook, "insert_rows", mock_insert_rows)

    new_episodes = load_episodes_logic(EPISODES_DATA)
    assert len(new_episodes) == 1
    assert new_episodes[0][0] == "https://example.com/episode2"

def mock_requests_get(*_args, **_kwargs) -> MagicMock:
    """
    Mocking the get method of requests.
    """
    mock_response = MagicMock()
    mock_response.content = b"mock audio content"
    return mock_response

def test_download_episodes_logic(monkeypatch, tmpdir) -> None:
    """
    Test that download_episodes_logic returns the correct number of downloaded episodes.
    """
    # Patching requests.get to use our mock function
    monkeypatch.setattr(requests, "get", mock_requests_get)

    # Using a temporary directory for the test
    temp_dir_str = str(tmpdir)
    monkeypatch.setattr("podcast_summary.EPISODE_FOLDER", temp_dir_str)

    downloaded_episodes = download_episodes_logic(EPISODES_DATA)
    assert len(downloaded_episodes) == 2
    assert downloaded_episodes[0]["link"] == "https://example.com/episode1"
    assert downloaded_episodes[1]["link"] == "https://example.com/episode2"

    # Checking if the files were "downloaded"
    assert os.path.isfile(os.path.join(temp_dir_str, "episode1.mp3"))
    assert os.path.isfile(os.path.join(temp_dir_str, "episode2.mp3"))
