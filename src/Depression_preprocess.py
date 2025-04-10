import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

def main():
    # Define file paths
    file_path = os.path.abspath('./data/welcome_raw_data.csv')
    processed_output_path = os.path.abspath('./data/processed_mh7a_data.csv')  # New file name
    
    welcome_data = pd.read_csv(file_path)
    
    # Step 2: Filter relevant features for predicting depression (MH7A)
    relevant_features = [
        'Age', 'Gender', 'Education', 'Household_Income', 'Subjective_Income', 'EMP_2010', 'age_var1', 
        'MH1', 'MH2A', 'MH2B', 'MH3B', 'MH6','MH5', 'MH7A'  
    ]
    
    df = welcome_data[relevant_features]
    
    # Step 3: Handle binary columns (Gender, MH6, MH7A)
    binary_columns = ['Gender', 'MH6', 'MH7A'] 
    for col in binary_columns:
        df[col] = df[col].replace({'': np.nan, ' ': np.nan, '  ': np.nan, '99': np.nan, 99: np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
    
    nominal_columns = ['EMP_2010']  
    df = pd.get_dummies(df, columns=nominal_columns, drop_first=False)
    
    # Step 5: Handle ordinal columns
    ordinal_columns = [
        'Age', 'Education', 'Household_Income', 'Subjective_Income', 'age_var1', 
        'MH1', 'MH2A', 'MH2B', 'MH3B', 'MH5'
    ]
    #target_columns = ['MH7A']  
    
    # Replace missing or invalid values in ordinal and target columns
    for col in ordinal_columns:
        df[col] = df[col].replace({'': np.nan, ' ': np.nan, '  ': np.nan, '99': np.nan, 99: np.nan})
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values for ordinal columns using mode
    for col in ordinal_columns:
        df[col] = df[col].fillna(df[col].mode()[0]).astype(int)
    
    # Apply ordinal encoding to ordinal columns
    ordinal_encoder = OrdinalEncoder()
    df[ordinal_columns] = ordinal_encoder.fit_transform(df[ordinal_columns])
    
    
    # Convert one-hot encoded nominal columns to integer (0 and 1)
    for col in df.columns:
        if col.startswith('EMP_2010'):
            df[col] = df[col].astype(int)
    # Step 6: Convert the MH7A class labels from 1 2 to 0/1
    df['MH7A'] = df['MH7A'].replace({1: 1, 2: 0})
    # MinMax scaling for the 'Age' column 
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    unique_values = df['MH7A'].unique()
    # Save the final processed DataFrame to CSV
    df.to_csv(processed_output_path, index=False)
    
    # Optionally, print the first few rows of the processed DataFrame
    print(df.head())

if __name__ == '__main__':
    main()
