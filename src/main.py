import os
import pandas as pd
from logg import logger
from Helpfulness_Training import random_forest, neural_network, xgboost

# Define the data path(for saving)
data_path = os.path.abspath('./data/processed_data_helpfulness.csv')

def insert_model_separator(model_name):
    """Creates a labeled separator just for CSV readability."""
    return pd.DataFrame([[f"----- {model_name} -----", "", "", "", ""]], 
                        columns=["Target", "Accuracy", "Precision", "Recall", "F1 Score"])

def main():
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded successfully from {data_path}. Shape: {df.shape}")
        
        # Define target columns
        target_columns = ['MH9A', 'MH9B', 'MH9C', 'MH9D', 'MH9E', 'MH9F', 'MH9G', 'MH9H']

        # Run models
        rf_results = random_forest(df, target_columns)
        nn_results = neural_network(df, target_columns)
        xgb_results = xgboost(df, target_columns)

        # Combine results with separators
        final_results = pd.concat([
            insert_model_separator("Random Forest"),
            rf_results,
            insert_model_separator("Neural Network"),
            nn_results,
            insert_model_separator("XGBoost"),
            xgb_results
        ], ignore_index=True)

        # Save to CSV
        os.makedirs("./results/main", exist_ok=True)
        output_path = "./results/main/Helpfulness_prediction.csv"
        final_results.to_csv(output_path, index=False)

        logger.info(f"\nResults saved to: {output_path}")
        logger.info("Model training and saving completed successfully.")

    except FileNotFoundError:
        logger.error(f"File not found: {data_path}. Please check the file path.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
