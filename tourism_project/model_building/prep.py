# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/dutta2arnab/tourism-package-prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)


print("Dataset loaded successfully.")

TARGET = "ProdTaken"
unique_id = "CustomerID"


# Identify the Numerical and Categorical Columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", cat_cols)
print("Numerical columns:", num_cols)

# Remove the target and unique id  from numerical columns
num_cols_no_target = [col for col in num_cols if col not in (TARGET,unique_id)]

print("Numerical columns (without target and unique id):", num_cols_no_target)


# Define the target variable for the classification task
target = TARGET

print(f"Target variable is {target}")



# Define predictor matrix (X) using selected numeric and categorical features

X = tourism_dataset[num_cols_no_target + cat_cols]

# Define target variable
y = tourism_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="dutta2arnab/tourism-package-prediction",
        repo_type="dataset",
    )
