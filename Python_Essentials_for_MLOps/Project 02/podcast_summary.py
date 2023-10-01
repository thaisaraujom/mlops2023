"""
This DAG downloads the latest episodes of the Marketplace 
podcast and stores them in a SQLite database.
"""
import os
import json
import logging
import requests
import xmltodict
import pendulum
import pandas as pd

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

from airflow.decorators import dag, task
from airflow.providers.sqlite.operators.sqlite import SQLExecuteQueryOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

INFO_EMOJI = "ℹ️"
ERROR_EMOJI = "❌"
DEBUG_EMOJI = "🔍"
SUCCESS_EMOJI = "✅"

PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "/opt/airflow/episodes"
FRAME_RATE = 16000

def create_database() -> SQLExecuteQueryOperator:
    """
    Create the SQLite database.

    Returns:
        SqliteOperator: Operator for creating the database.
    """
    return SQLExecuteQueryOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        conn_id="podcasts"
    )

def fetch_episodes() -> list:
    """
    Fetch the latest episodes from the podcast URL.

    This function is a general utility for fetching the latest 
    episodes from a specified podcast URL.
    
    Returns:
        list: List of the latest episodes.
    """
    try:
        data = requests.get(PODCAST_URL, timeout=15)
        feed = xmltodict.parse(data.text)
        episodes = feed["rss"]["channel"]["item"]
        logging.info("%s Found %d episodes.", INFO_EMOJI, len(episodes))
        return episodes
    except requests.RequestException as request_error:
        logging.error("%s Error fetching episodes: %s", ERROR_EMOJI, request_error)
        raise
    except Exception as general_error:
        logging.error(
            "%s Unexpected error during get_episodes: %s", ERROR_EMOJI, general_error
        )
        raise

@task()
def get_episodes() -> list:
    """
    Airflow task to fetch the latest episodes from the podcast URL.

    This task is meant to be used within an Airflow DAG to 
    fetch the latest episodes from a specified podcast URL.
    
    Returns:
        list: List of the latest episodes.
    """
    return fetch_episodes()

def load_episodes_logic(episodes: list) -> list:
    """
    Load new episodes to the SQLite database.
    
    This function is a general utility for loading new episodes 
    to the SQLite database from a list of episodes.
    
    Args:
        episodes (list): List of the latest episodes.

    Returns:
        list: List of new episodes added to the database.
    """
    hook = SqliteHook(sqlite_conn_id="podcasts")
    stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
    new_episodes = []
    for episode in episodes:
        if episode["link"] not in stored_episodes["link"].values:
            filename = f"{episode['link'].split('/')[-1]}.mp3"
            new_episodes.append([episode["link"],
                                    episode["title"],
                                    episode["pubDate"],
                                    episode["description"], filename])
    target_fields = ["link", "title", "published", "description", "filename"]
    hook.insert_rows(table='episodes',
                        rows=new_episodes,
                        target_fields=target_fields)
    return new_episodes

@task()
def load_episodes(episodes: list) -> list:
    """
    Airflow task to load new episodes to the SQLite database.
    
    This task is meant to be used within an Airflow DAG to load new episodes to 
    the SQLite database from a list of episodes.
    
    Args:
        episodes (list): List of the latest episodes.

    Returns:
        list: List of new episodes added to the database.
    """
    return load_episodes_logic(episodes)

def download_episodes_logic(episodes: list) -> list:
    """
    Download the specified podcast episodes.
    
    This function is a general utility for downloading podcast 
    episodes from a list of episodes.
    
    Args:
        episodes (list): List of episodes to download.

    Returns:
        list: List of dictionaries with 'link' and 'filename' keys for the downloaded episodes.
    """
    audio_files = []
    for episode in episodes:
        try:
            name_end = episode["link"].split('/')[-1]
            filename = f"{name_end}.mp3"
            audio_path = os.path.join(EPISODE_FOLDER, filename)
            if not os.path.exists(audio_path):
                logging.info("%s Downloading %s", INFO_EMOJI, filename)
                audio = requests.get(episode["enclosure"]["@url"], timeout=15)
                with open(audio_path, "wb+") as file:
                    file.write(audio.content)
            audio_files.append({
                "link": episode["link"],
                "filename": filename
            })
        except requests.RequestException as request_error:
            logging.error("%s Error downloading episode %s: %s",
                            ERROR_EMOJI, episode['link'], request_error)
            raise

        except IOError as file_error:
            logging.error("%s File error with %s: %s",
                            ERROR_EMOJI, filename, file_error)
            raise

        except Exception as general_error:
            logging.error("%s Unexpected error during download_episodes: %s",
                            ERROR_EMOJI, general_error)
            raise

    return audio_files

@task()
def download_episodes(episodes: list) -> list:
    """
    Airflow task to download the specified podcast episodes.
    
    This task is meant to be used within an Airflow DAG to 
    download podcast episodes from a list of episodes.
    
    Args:
        episodes (list): List of episodes to download.

    Returns:
        list: List of dictionaries with 'link' and 'filename' keys for the downloaded episodes.
    """
    return download_episodes_logic(episodes)

def fetch_untranscribed_episodes(hook: SqliteHook) -> pd.DataFrame:
    """
    Fetch episodes that haven't been transcribed yet.

    Args:
        hook (SqliteHook): SqliteHook instance for database operations.

    Returns:
        pd.DataFrame: Dataframe of untranscribed episodes.
    """
    query = (
        "SELECT * "
        "FROM episodes "
        "WHERE transcript IS NULL;"
    )
    return hook.get_pandas_df(query)

def initialize_transcription_model() -> Model:
    """
    Initialize the transcription model.

    Returns:
        Model: Vosk model instance for transcription.
    """
    return Model(model_name="vosk-model-en-us-0.22-lgraph")

def transcribe_audio(row: pd.Series, rec: KaldiRecognizer) -> str:
    """
    Transcribe a specific audio file.

    Args:
        row (pd.Series): Data row of the episode.
        rec (KaldiRecognizer): KaldiRecognizer instance.

    Returns:
        str: Transcribed text.
    """
    filepath = os.path.join(EPISODE_FOLDER, row["filename"])
    mp3 = AudioSegment.from_mp3(filepath).set_channels(1).set_frame_rate(FRAME_RATE)

    step = 20000
    transcript = ""
    for i in range(0, len(mp3), step):
        logging.debug("%s Progress: %f", DEBUG_EMOJI, i / len(mp3))
        segment = mp3[i:i + step]
        rec.AcceptWaveform(segment.raw_data)
        result = rec.Result()
        text = json.loads(result)["text"]
        transcript += text
    return transcript

def store_transcript(hook: SqliteHook, link: str, transcript: str) -> None:
    """
    Store the transcribed text into the database.

    Args:
        hook (SqliteHook): SqliteHook instance for database operations.
        link (str): Link of the episode.
        transcript (str): Transcribed text of the episode.
    
    Returns:
        None
    """
    hook.insert_rows(
        table='episodes',
        rows=[[link, transcript]],
        target_fields=["link", "transcript"],
        replace=True
    )

@task()
def speech_to_text() -> None:
    """
    Transcribe the audio content of the episodes to text.

    Returns:
        None
    """
    hook = SqliteHook(sqlite_conn_id="podcasts")
    untranscribed_episodes = fetch_untranscribed_episodes(hook)

    model = initialize_transcription_model()
    rec = KaldiRecognizer(model, FRAME_RATE)
    rec.SetWords(True)

    for _, row in untranscribed_episodes.iterrows():
        logging.info("%s Transcribing %s", INFO_EMOJI, row['filename'])
        transcript = transcribe_audio(row, rec)

        store_transcript(hook, row["link"], transcript)

@dag(
    dag_id='podcast_summary',
    schedule="@daily",
    start_date=pendulum.datetime(2023, 9, 29),
    catchup=False,
)

def podcast_summary() -> None:
    """
    DAG definition for the podcast summary.

    Returns:
        None
    """
    create_database_task = create_database()

    podcast_episodes = get_episodes()
    create_database_task.set_downstream(podcast_episodes)

    load_episodes(podcast_episodes)
    download_episodes(podcast_episodes)

    # Uncomment this to try speech to text (may not work)
    # speech_to_text()

podcast_summary()
