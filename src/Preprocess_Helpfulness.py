import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

def main():
    # Define file paths
    file_path = os.path.abspath('./data/welcome_raw_data.csv')
    filtered_output_path = os.path.abspath('./data/helpfulness_score.csv')
    processed_output_path = os.path.abspath('./data/processed_data.csv')
    
    # Load the row data file into a DataFrame
    welcome_data = pd.read_csv(file_path)
    
    # Keep only yes respondents for depression
    welcome_data = welcome_data[welcome_data['MH7A'] == 1]
    
    # Step 1: Filter relevant features
    relevant_features = [
        'Global11Regions', 'W1', 'W4', 'W5A', 'W5B', 'W5C', 'W5D', 'W5E', 'W5F', 'W5G',
        'W6', 'W7A', 'MH2A', 'MH2B', 'MH3B', 'W10', 'W11A', 'wbi', 'Age', 'age_mh', 
        'Gender', 'Education', 'Household_Income', 'Subjective_Income', 'EMP_2010',
        'MH1', 'MH4B', 'MH5', 'age_var1', 'MH6', 'MH7C',
        'MH9A', 'MH9B', 'MH9C', 'MH9D', 'MH9E', 'MH9F', 'MH9G', 'MH9H'
    ]
    filtered_data = welcome_data[relevant_features]
    filtered_data.to_csv(filtered_output_path, index=False)
    
    df = filtered_data.copy()
    
    # Step 2: Handle binary columns (Gender, MH6, MH7C)
    binary_columns = ['Gender', 'MH6', 'MH7C']
    for col in binary_columns:
        df[col] = df[col].replace({'': np.nan, ' ': np.nan, '  ': np.nan, '99': np.nan, 99: np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
    
    # Step 3: One-hot encode nominal columns (Global11Regions, EMP_2010, W10, W11A)
    nominal_columns = ['Global11Regions', 'EMP_2010', 'W10', 'W11A']
    df = pd.get_dummies(df, columns=nominal_columns, drop_first=False)
    
    # Step 4: Handle ordinal columns and target columns
    ordinal_columns = [
        'Subjective_Income', 'Education', 'Household_Income', 'MH5', 'age_mh',
        'W1', 'W4', 'W5A', 'W5B', 'W5C', 'W5D', 'W5E', 'W5F', 'W5G',
        'W6', 'W7A', 'MH2A', 'wbi', 'age_var1', 'MH1', 'MH2B', 'MH3B', 'MH4B'
    ]
    target_columns = ['MH9A', 'MH9B', 'MH9C', 'MH9D', 'MH9E', 'MH9F', 'MH9G', 'MH9H']
    
    # Replace missing or invalid values in ordinal and target columns
    for col in ordinal_columns + target_columns:
        df[col] = df[col].replace({'': np.nan, ' ': np.nan, '  ': np.nan, '99': np.nan, 99: np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values for ordinal columns using mode
    for col in ordinal_columns:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
    
    # Apply ordinal encoding
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_columns + target_columns] = ordinal_encoder.fit_transform(df[ordinal_columns + target_columns])
    
    # Fill missing target values with 99 and convert to integer(since we will chooseonly three of the values it's okay to stay 99 as it is)
    for col in target_columns:
        df[col] = df[col].fillna(99).astype(int)
    
    # Ensure integer encoding for ordinal and target columns
    df[ordinal_columns + target_columns] = df[ordinal_columns + target_columns].astype(int)
    
    # Convert one-hot encoded nominal columns to integer (0 and 1)
    for col in df.columns:
        if col.startswith('EMP_2010') or col.startswith('Global11Regions') or col.startswith('W10') or col.startswith('W11A'):
            df[col] = df[col].astype(int)
    
    # MinMax scaling for the 'Age' column
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    
    # Save the final processed DataFrame to CSV
    df.to_csv(processed_output_path, index=False)

if __name__ == '__main__':
    main()
