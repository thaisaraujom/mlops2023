# 🩺 Project 03 - Predicting Heart Disease
The project is based on the _Guided Project: Predicting Heart Disease_ available on [Dataquest](https://app.dataquest.io/). The objective is to modify the project by applying the concepts learned in the MLOps course. This project follows a machine learning pipeline, where several critical steps are carried out to predict the occurrence of heart diseases using a provided dataset. The `load_data` function automates data loading from a CSV file, while `clean_data` ensures data quality. Through `log_data_info`, basic dataset information is logged, and `plot_categorical_counts` alongside `plot_correlation_heatmap` aid in visual data exploration. Model training is handled by `train_knn` and `grid_search_knn`, which train a K-Nearest Neighbors classifier and optimize hyperparameters respectively. The `evaluate_model` function assesses the model's performance on a test set, and `log_sex_distribution` provides insight into data distribution by sex. The `main` function orchestrates the execution of these functions, ensuring a logical flow from data ingestion to model evaluation, embodying MLOps practices like logging and modularization for better maintainability and monitoring in a production setting.

## 🔧 Requirements/Technologies
- Python 3.8+
- Python packages: 
   - pylint
   - astroid
   - pytest
   - pandas
   - matplotlib
   - seaborn
   - scikit-learn
 
   You can find the full list of requirements in the `requirements.txt` file.

## 🚀 Installation Instructions
1. Ensure you have Python version 3.8+ installed.
2. Clone the repository: 
   ```
   git clone https://github.com/thaisaraujom/mlops2023.git
   ```
3. Navigate to the `Python_Essentials_for_MLOps` directory and then to the `Project 03` directory.
4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/).
5. Install the required libraries: 
   ```
   pip install -r requirements.txt
   ``` 
6. Run the project:
   ```
   python heart_disease_prediction.py
   ```
7. To run the tests, run the following command:
   ```
    pytest 
   ```
8. To run the linter, run the following command:
   ```
    pylint heart_disease_prediction.py
   ```
9. Alternatively, you can use **GitHub Codespaces** for a quick setup:
   - In the main repository page, click on the `Code` button and select `Codespaces`.
   - From the dropdown, select `Open with Codespaces` or `Create codespace on main`.
   - This will initiate a Codespace environment where you can code directly in the browser without local setup.

## 📚 References
- [Jupyter Notebook for the original project (Dataquest)](https://github.com/dataquestio/solutions/blob/master/Mission740Solutions.ipynb)