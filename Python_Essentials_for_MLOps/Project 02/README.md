# 🎙️ Project 02 - An Airflow Data Pipeline to Download Podcasts
The project, based on the _Build an Airflow Data Pipeline to Download Podcasts_ from [Dataquest](https://app.dataquest.io/), aims to enhance the original by incorporating MLOps concepts. The code defines an Airflow DAG that automates the downloading of recent Marketplace podcast episodes and stores their information in a SQLite database. It encompasses tasks for database creation, episode fetching, data loading, and audio file downloading.

## 🔧 Requirements/Technologies
- Airflow 2.3+
- Python 3.8+
- Python packages: 
   - pylint
   - astroid
   - pytest
   - requests
   - pandas
   - vosk
   - pydub
   - xmltodict
   - sqlite3
 
   You can find the full list of requirements in the `requirements.txt` file.

## 🚀 Installation Instructions
1. Ensure you have Python version 3.8+ and Airflow 2.3+ installed. You can find instructions on how to install Airflow [here](https://airflow.apache.org/docs/apache-airflow/stable/start.html). Remember to configure the path to the `AIRFLOW_HOME` environment variable and the `airflow.cfg` file as instructed in the documentation to define the Airflow home directory and path to the dags folder.

2. Install sqlite3 following the instructions [here](https://www.sqlite.org/download.html).

3. Clone the repository: 
   ```
   git clone https://github.com/thaisaraujom/mlops2023.git
   ```

4. Navigate to the `Python_Essentials_for_MLOps` directory and then to the `Project 02` directory.

5. Install the required libraries: 
   ```
   pip install -r requirements.txt
   ``` 

6. Create a folder named `episodes` in the `Project 02` directory and change the `EPISODE_FOLDER` variable in the `podcast_summary.py` file to the path of the folder you just created.

7. You will need to create a `episodes.db` file in the `Project 02` directory. To do so, run the following command:
   ```
   sqlite3 episodes.db
   ```

8. Create a connection to the database by running the following command:
   ```
   airflow connections add 'podcasts' --conn-type 'sqlite' --conn-host '/path/episodes.db'
   ```
   replacing `/path/episodes.db` with the path to the `episodes.db` file you just created.

9. Run the project in airflow UI manually pressing the `play` button in the `podcast_summary` DAG or define a schedule for the DAG to run automatically for a given interval.

10. To run the tests, run the following command:
      ```
      pytest
      ```

11. To run the linter, run the following command:
      ```
      pylint podcast_summary.py
      ```

## 📚 References
- [Code for the original project (Dataquest)](https://github.com/dataquestio/project-walkthroughs/tree/master/podcast_summary)