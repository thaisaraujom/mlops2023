# 🎬 Project 01 - A Movie Recommendation System in Python
The project is based on the _Build a Movie Recommendation System in Python_, a Portfolio Project available on [Dataquest](https://app.dataquest.io/). The goal is to modify the project by applying the concepts learned in the MLOps course. The code implements a movie recommendation system using a content-based filtering technique, comparing the similarity between movies using TF-IDF and cosine similarity. Upon execution, the code downloads a dataset, processes and searches for movies similar to a provided title, delivering a list of recommendations to the user.

## 🔧 Requirements/Technologies
- Python 3.8+
- Python packages: 
   - pylint
   - astroid
   - termcolor
   - tabulate
   - tqdm
   - pytest
   - requests
   - numpy
   - pandas
 
   You can find the full list of requirements in the `requirements.txt` file.

## 🚀 Installation Instructions
1. Ensure you have Python version 3.8+ installed.
2. Clone the repository: 
   ```
   git clone https://github.com/thaisaraujom/mlops2023.git
   ```
3. Navigate to the `Python_Essentials_for_MLOps` directory and then to the `Project 01` directory.
4. Install the required libraries: 
   ```
   pip install -r requirements.txt
   ``` 
5. Run the project:
   ```
   python movie_recommendations.py -t "Movie Title"
   ```
   replacing `Movie Title` with the title of the movie you want to get recommendations for.
6. To run the tests, run the following command:
   ```
    pytest 
   ```
7. To run the linter, run the following command:
   ```
    pylint movie_recommendations.py
   ```
8. Alternatively, you can use **GitHub Codespaces** for a quick setup:
   - In the main repository page, click on the `Code` button and select `Codespaces`.
   - From the dropdown, select `Open with Codespaces` or `Create codespace on main`.
   - This will initiate a Codespace environment where you can code directly in the browser without local setup.

## 📚 References
- [Jupyter Notebook for the original project (Dataquest)](https://github.com/dataquestio/project-walkthroughs/blob/master/movie_recs/movie_recommendations.ipynb)