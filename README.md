# Predicting the Helpfulness of Mental Health Treatment Methods Using Machine Learning

## Project Overview
This project aims to predict the perceived helpfulness of various mental health treatment methods using machine learning. It leverages data from the **Wellcome Global Monitor 2020 Mental Health data** to understand how different treatment approaches (e.g., talking to professionals, medication, lifestyle changes) are perceived by individuals with mental health issues such as anxiety or depression. The primary focus is to predict the helpfulness of these treatments, which can aid in personalizing mental health care and improving clinical decision-making.

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

## Folder Structure
The repository is structured as follows:

/data /raw_welcomedata # Raw Wellcome Global Monitor Data /processed_data # Processed data for helpfulness prediction /processed_depression_data # Processed data for depression prediction (exploratory)

/notebooks distribution.ipynb # Data distribution analysis and visualization result_analysis.ipynb # Result visualization and analysis cluster.ipynb # Clustering analysis using unsupervised methods

/src helpfulness_training.py # Script to train machine learning models for helpfulness prediction helpfulness_preprocess.py # Preprocessing script for helpfulness prediction data depression_preprocess.py # Preprocessing script for depression prediction data main.py # Main script to initiate helpfulness prediction depression_prediction.py # Script for depression prediction analysis back_prediction.py # Predict demographic variables and analyze results logg.py # Logger for tracking model training and results
