import subprocess

# Run a terminal command (e.g., checking for Python version)
subprocess.run(["python", "--version"])

# Continue with your script
def main(args):
    # Your script logic here
    pass



# Import necessary libraries
import argparse
import glob
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Enable MLflow autologging
mlflow.sklearn.autolog()

# Define functions
def main(args):
    # Read data
    df = get_csvs_df(args.training_data)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)

def get_csvs_df(path):
    print(f"Checking path: {os.path.abspath(path)}")  # Print absolute path for debugging
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    # Check for missing values
    if df.isnull().values.any():
        print("Warning: Missing values found in the dataset.")
        # Handle missing values, e.g., using mean imputation
        df = df.fillna(df.mean())

    # Convert integer columns to float to handle potential missing values
    df = df.apply(lambda x: x.astype(float) if x.dtype == 'int' else x)

    # Assume the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # Train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)

def parse_args():
    # Setup argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str, required=True, help="Path to training data directory")
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01, help="Regularization rate")

    # Parse args
    return parser.parse_args()

# Run script
if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Run main function
    main(args)
