# Predicting the Helpfulness of Mental Health Treatment Methods Using Machine Learning

## Project Overview
This project aims to predict the perceived helpfulness of various mental health treatment methods using machine learning. It leverages data from the [Wellcome Global Monitor 2020 Mental Health data](https://wellcome.org/reports/wellcome-global-monitor-mental-health/2020) to understand how different treatment approaches (e.g., talking to professionals, medication, lifestyle changes) are perceived by individuals with mental health issues such as anxiety or depression. The primary focus is to predict the helpfulness of these treatments, which can aid in personalizing mental health care and improving clinical decision-making.
We also included the reverse prediction of demographic variables and prediction of depression risk to understand how the data is related.
## Data Description
The data used in this project comes from the **Wellcome Global Monitor (WGM) 2020 Mental Health Module**. It contains survey responses from individuals who self-reported having experienced anxiety or depression. The key features of the dataset include:

- **Target Variables (MH9A–MH9H)**: The self-reported perceived helpfulness of eight distinct treatment methods (rated on a 3-point scale: 1 = Very helpful, 2 = Somewhat helpful, 3 = Not helpful):
  - **MH9A**: Talking to a mental health professional
  - **MH9B**: Engaging in religious/spiritual activities
  - **MH9C**: Talking to friends or family
  - **MH9D**: Taking prescribed medication
  - **MH9E**: Improving healthy lifestyle behaviors
  - **MH9F**: Changing work situation
  - **MH9G**: Changing personal relationships
  - **MH9H**: Spending time in nature/outdoors

- **Predictor Features**: These include demographic and mental health-related data like age, gender, education, income level, employment status, experience with mental health issues, and beliefs about mental health.

### Folder Structure
- **`data/`**: Contains datasets for the project.
  - `welcome_raw_data.csv`: Original raw dataset.
  - `processed_depression.csv`: Preprocessed data for depression prediction tasks.
  - `processed_data_helpfulness.csv`: Preprocessed data for helpfulness prediction tasks.
  - `clustered_data.csv`: Data after applying clustering .

- **`notebooks/`**: Jupyter notebooks for analysis and visualization.
  - `Distribution.ipynb`: Data distribution and demographic analysis.
  - `result_analysis.ipynb`: Model evaluation and analysis of results.
  - `Unsupervised_cluster.ipynb`: Unsupervised clustering and visualization.

- **`src/`**: Core scripts for processing, model training, and utilities.
  - `Demographic_prediction.py`: Script for predicting demographic-based outcomes.
  - `Depression_Prediction.py`: Handles depression model training and prediction.
  - `Depression_preprocess.py`: Data preprocessing for depression task.
  - `Helpfulness_Training.py`: Training script for helpfulness prediction.
  - `Preprocess_Helpfulness.py`: Data cleaning and preprocessing for helpfulness task.
  - `main.py`: Main execution script for starting the helpfulness training.
  - `logg.py`: Logging setup for tracking pipeline progress.

- **`results/`**: Final results and documentation.
-  `main/Helpfulness_prediction.csv`: Helpfulness model predictions.
  - `Depression_prediction_results.csv`: Predictions from depression model.
  - `demographic_prediction_result.csv`: Results from demographic-based models.
  - `Predicting the Helpfulness of Mental Health Treatment Methods Using ML.docx`: Project report.
  
- **`Test/`** : Unit tests for core functionalities.
  - `test.py`: Test cases for data loading, preprocessing, and evaluation.
