import os
import pandas as pd
import numpy as np
import importlib.util
import subprocess

def run_preprocessing():
    print("🔄 Running preprocessing script...")
    result = subprocess.run(["python", "preprocess_helpfulness.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Preprocessing failed.")
        print(result.stderr)
    else:
        print("✅ Preprocessing completed.")
        print(result.stdout)

def load_and_check_processed_data(path):
    if not os.path.exists(path):
        print(f"❌ Processed file not found at {path}")
        return None

    df = pd.read_csv(path)
    print("✅ Processed data loaded successfully.")

    # Basic checks
    print(f"Shape: {df.shape}")
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")

    if missing == 0:
        print("✅ No missing values detected.")
    else:
        print("⚠️ Warning: Some missing values detected.")

    print("Columns:", df.columns.tolist()[:10], "...")
    return df

def evaluate_dummy(df):
    print("\n🧪 Running basic dummy evaluation...")

    # Basic stat check: Target columns like MH9A–MH9H should be present
    target_cols = [col for col in df.columns if col.startswith('MH9')]
    if not target_cols:
        print("❌ No target columns (MH9A-H) found.")
        return
    
    print("✅ Target columns:", target_cols)

    # Just show value counts for one target as a placeholder
    print(f"\nValue counts for {target_cols[0]}:\n", df[target_cols[0]].value_counts())

    print("\n✅ Evaluation sanity check complete.")

def main():
    processed_path = os.path.abspath('./data/processed_data_helpfulness.csv')
    
    run_preprocessing()
    
    df = load_and_check_processed_data(processed_path)
    
    if df is not None:
        evaluate_dummy(df)

if __name__ == "__main__":
    main()
