import pandas as pd
import os
import json
import random
import codecs
import logging
import numpy as np
from sklearn.externals import joblib
from typing import List
from google.cloud import storage, bigquery


def download_from_gcs(bucket_dir_name: str, file_name: str):
    PROJECT_ID = "wantedly-individual-shu"
    GCS_BUCKET_NAME = "booking-challenge-2021-wantedly"

    client = storage.Client(project=PROJECT_ID)
    bucket = client.get_bucket(GCS_BUCKET_NAME)

    blob = storage.Blob(
        os.path.join(bucket_dir_name, file_name),
        bucket
    )
    content = blob.download_as_string()
    print(f"Downloading {file_name} from {blob.path}")

    return content


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("=== dataframe dtypes before optimization ===")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    print("============================================")
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    # Float16 is not implemented in the Feather format yet.
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    print("=== dataframe dtypes after optimization  ===")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    print("============================================")

    return df
