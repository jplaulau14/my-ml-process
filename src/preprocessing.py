import pandas as pd
import numpy as np
from collections import Counter
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from typing import List, Dict, Tuple, Optional
import logging
import sys
import os
import datetime
from src.paths import get_data_path

def rename_columns(df: pd.DataFrame, col_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Rename columns of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to rename columns of.
    col_map : Dict[str, str], optional
        Dictionary mapping old column names to new column names, by default None

    Returns
    -------
    df : pd.DataFrame
        Dataframe with renamed columns.
    """

    if col_map is None:
        # If col_map is None, replace whitespace with underscores and lowercase all column names
        logging.info('No column map specified. Renaming columns to lowercase and replacing whitespace with underscores')
        col_map = {col: col.lower().replace(" ", "_") for col in df.columns}
    else:    
        logging.info(f'Renaming columns using column map: {col_map}')
        df = df.rename(columns=col_map)
    
    return df

# time_type is an optional list of time dtypes to extract
def time_processing(df: pd.DataFrame, time_column: str = 'time', time_type: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract time features from a time column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to extract time features from.
    time_column : str, optional
        Name of the time column, by default 'time'
    time_type : List[str], optional
        List of time types to extract, by default None

    Returns
    -------
    df : pd.DataFrame
        Dataframe with extracted time features.
    """
    # Create accepted time types
    logging.info(f'Extracting time features from {time_column}')
    accepted_time_types = ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'dayofyear', 'quarter']

    # Convert column to datetime
    df[time_column] = pd.to_datetime(df[time_column])

    # Extract time features that are in time_type
    if time_type is None:
        logging.info('No time types specified. No time features will be extracted.')
        return df
    else:
        for time in time_type:
            if time in accepted_time_types:
                df[time] = getattr(df[time_column].dt, time)
            else:
                logging.warning(f'{time} is not a valid time type. Accepted time types are {accepted_time_types}')
    df.drop(time_column, axis=1, inplace=True)
    logging.info(f'Extracted time features: {time_type}')
    return df

def drop_null_cols(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Drop columns from a dataframe that have null value count greater than the threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to drop columns from.
    threshold : int
        Threshold for dropping columns.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with dropped columns.
    """
    logging.info(f'Dropping columns with null value count greater than {threshold}')
    # Get columns with null value count greater than threshold
    null_cols = df.columns[df.isnull().sum() > threshold]
    # Drop columns
    df = df.drop(null_cols, axis=1)
    logging.info(f'Dropped columns: {null_cols}')
    return df

def ordinal_encoder(df: pd.DataFrame, features: pd.DataFrame.columns) -> pd.DataFrame:
    """
    Perform ordinal encoding on the given features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to perform ordinal encoding on.
    features : pd.DataFrame.columns
        Features to perform ordinal encoding on.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with encoded features.
    """
    logging.info(f'Performing ordinal encoding on {features}')
    for feature in features:
        feature_val = list(np.arange(df[feature].nunique()))
        feature_key = list(df[feature].sort_values().unique())
        feature_dict = dict(zip(feature_key, feature_val))
        df[feature] = df[feature].map(feature_dict)
    logging.info(f'Performed ordinal encoding on {features}')
    return df

def knn_impute(df: pd.DataFrame, n_neighbors: Optional[int] = 5) -> pd.DataFrame:
    """
    Perform KNN imputation on a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to perform KNN imputation on.
    n_neighbors : int, optional
        Number of neighbors to use for imputation, by default 5

    Returns
    -------
    df : pd.DataFrame
        Dataframe with imputed values.
    """
    logging.info(f'Performing KNN imputation on dataframe with {n_neighbors} neighbors')
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    logging.info(f'Performed KNN imputation on dataframe')
    return df

def smote(df: pd.DataFrame, target: str, sampling_strategy: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Perform SMOTE oversampling on a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to perform SMOTE oversampling on.
    target : str
        Name of the target column.
    sampling_strategy : Dict[str, int], optional
        Dictionary specifying the number of samples to generate for each class, by default None

    Returns
    -------
    df : pd.DataFrame
        Dataframe with oversampled data.
    """
    logging.info(f'Performing SMOTE oversampling on dataframe')
    sm = SMOTE(sampling_strategy=sampling_strategy)
    X = df.drop(target, axis=1)
    y = df[target]
    logging.info('Class distribution before oversampling:')
    logging.info('-------------------------------------')
    counter = Counter(y)
    for k,v in counter.items():
        per = v / len(y) * 100
        logging.info(f'Class={k}, n={v} ({per:.3f}%)')
    X_res, y_res = sm.fit_resample(X, y)
    logging.info('Class distribution after oversampling:')
    logging.info('-------------------------------------')
    counter = Counter(y_res)
    for k,v in counter.items():
        per = v / len(y_res) * 100
        logging.info(f'Class={k}, n={v} ({per:.3f}%)')
    df = pd.concat([X_res, y_res], axis=1)
    logging.info(f'Performed SMOTE oversampling on dataframe')
    logging.info(f'New dataframe shape: {df.shape}')
    return df

def preprocess_data_versioning(df: pd.DataFrame) -> None:
    """
    Create a data versioning system by saving versions to preprocessed data directory

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    """
    data_path = get_data_path()

    # Check if preprocessed data directory exists inside data_path
    if not os.path.exists(os.path.join(data_path, 'preprocessed')):
        os.makedirs(os.path.join(data_path, 'preprocessed'))
    # Count the number of files in the preprocessed data directory
    num_files = len(os.listdir(os.path.join(data_path, 'preprocessed')))
    # Save the current version of the preprocessed data
    df.to_csv(os.path.join(data_path, 'preprocessed', f'data_v{num_files + 1}.csv'), index=False)

def preprocess(df: pd.DataFrame, threshold: int, time_column: str, time_type: List[str], target: str, n_neighbors: int, sampling_strategy: Optional[Dict[str, int]] = None, col_map: Optional[Dict[str, str]] = None,) -> pd.DataFrame:
    """
    Perform preprocessing on a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to perform preprocessing on.
    threshold : int
        Threshold for dropping columns.
    time_column : str
        Name of the column containing time data.
    time_type : List[str]
        List of time types to extract.
    target : str
        Name of the target column.
    n_neighbors : int
        Number of neighbors to use for KNN imputation.
    sampling_strategy : Dict[str, int], optional
        Dictionary specifying the number of samples to generate for each class, by default None
    col_map : Dict[str, str], optional
        Dictionary mapping column names to new column names, by default None

    Returns
    -------
    df : pd.DataFrame
        Dataframe with preprocessed data.
    """
    logging.info('Performing preprocessing')
    # Rename columns
    df = rename_columns(df, col_map)
    # Drop columns with null value count greater than threshold
    df = drop_null_cols(df, threshold)
    # Extract time features
    df = time_processing(df, time_column, time_type)
    # Perform ordinal encoding
    df = ordinal_encoder(df, df.columns)
    # Perform KNN imputation
    df = knn_impute(df, n_neighbors)
    logging.info('Performed preprocessing')
    # Save preprocessed data
    preprocess_data_versioning(df)
    logging.info('Saved preprocessed data')
    return df
    