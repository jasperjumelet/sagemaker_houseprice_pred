from __future__ import print_function

import argparse
import os
import warnings
import logging

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import  ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.exceptions import DataConversionWarning
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    logger.info("Received arguments {}".format(args))

    # input_data_path = os.path.join("/opt/ml/processing/input", "train.csv")
    input_data_path = "train.csv"

    logger.info("Reading input data from {}".format(input_data_path))

    df = pd.read_csv(input_data_path)


    split_ratio = args.train_test_split_ratio

    print("Splitting data into train and test sets with ratio {}".format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("SalePrice", axis=1), df["SalePrice"], test_size=split_ratio, random_state=0
    )

    num_cols = df.drop(["SalePrice"], axis=1).select_dtypes(exclude="object").columns.tolist()
    cat_cols = df.drop(["SalePrice"], axis=1).select_dtypes(include="object").columns.tolist()

    all_features = num_cols + cat_cols

    num_transformer = Pipeline(steps=[('num_imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                    ('scaler', StandardScaler())]) #change strategy to see how permance changes

    cat_transformer = Pipeline(steps=[('cat_imputer', SimpleImputer(strategy='constant', fill_value="NA"))])

    preprocess = ColumnTransformer(
        transformers=[
            ('num_transformations', num_transformer, num_cols),
            ('cat_transformations', cat_transformer, cat_cols)
        ]
    )
    target_enc = TargetEncoder()
    train_features = preprocess.fit_transform(X_train)
    train_features = target_enc.fit_transform(train_features, y_train)


    # Save the transformers
    joblib.dump(preprocess, 'preprocess_pipeline_houseprice.pkl')
    joblib.dump(preprocess, 'target_encoder_houseprice.pkl')

    # Note that you can easy load the files again later in inference script
    # preprocess = joblib.load('preprocess_pipeline_houseprice.pkl')
    # target_enc = joblib.load('target_encoder_houseprice.pkl')

    test_features = preprocess.transform(X_test)
    test_features = target_enc.transform(test_features)

    print(train_features.head())

    print("X Train data shape after preprocessing: {}".format(train_features.shape))
    print("Y Train data shape after preprocessing: {}".format(y_train.shape))
    print("Test data shape after preprocessing: {}".format(test_features.shape))
    print("Y test data shape after preprocessing: {}".format(y_test.shape))

