from __future__ import print_function

import argparse
import os
import warnings
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import  ColumnTransformer
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.exceptions import DataConversionWarning
from sklearn.pipeline import Pipeline

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--train-test-split-ration', type=float, default=0.3)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    logger.info("Received arguments {}".format(args))

    # input_data_path = os.path.join("/opt/ml/processing/input", "train.csv")
    input_data_path = "train.csv"

    logger.info("Reading input data from {}".format(input_data_path))

    df = pd.read_csv(input_data_path)
    num_cols = df.drop(["Id", "SalePrice"], axis=1).select_dtypes(exclude="object").columns.tolist()
    cat_cols = df.drop(["Id", "SalePrice"], axis=1).select_dtypes(include="object").columns.tolist()

    all_features = num_cols + cat_cols

    num_transformer = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy='constant', fill_value=0)),
            ("scaler", StandardScaler())
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("cat_imputer", SimpleImputer(strategy='constant', fill_value="NA")),
            ("encoder", TargetEncoder())
        ]
    )   

    # Combine the transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num_transformations", num_transformer, num_cols),
            # ("cat_transformations", cat_transformer, cat_cols)
        ]
    )

    X = preprocessor.fit_transform(df[num_cols])
    y = df["SalePrice"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X)
    # Store train/test splits into files
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)