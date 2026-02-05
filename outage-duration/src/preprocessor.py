"""
preprocessor.py - data cleaning and feature engineering

takes the raw EAGLE-I and NOAA data and transforms it into
something we can actually feed into the model. handles merging,
feature extraction, and all that fun stuff
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def extractTemporalFeatures(df: pd.DataFrame, timestampCol: str = 'start_time') -> pd.DataFrame:
    """
    pulls out time-based features from a timestamp column

    extracts year, month, day, hour, dayofweek as specified in the PRD

    Args:
        df: dataframe with a timestamp column
        timestampCol: name of the column to extract from

    Returns:
        dataframe with new temporal feature columns added
    """
    df = df.copy()

    if timestampCol not in df.columns:
        raise ValueError(f"column {timestampCol} not found in dataframe")

    ts = pd.to_datetime(df[timestampCol])

    df['year'] = ts.dt.year
    df['month'] = ts.dt.month
    df['day'] = ts.dt.day
    df['hour'] = ts.dt.hour
    df['dayofweek'] = ts.dt.dayofweek

    return df


def mergeWeatherData(outageDf: pd.DataFrame, weatherDf: pd.DataFrame,
                     on: List[str] = None) -> pd.DataFrame:
    """
    merges the outage data with weather data

    tries to join on common columns like fips_code and date.
    you might need to adjust the merge keys depending on how
    the actual data is structured

    Args:
        outageDf: the EAGLE-I outage dataframe
        weatherDf: the NOAA weather dataframe
        on: list of columns to merge on (if None, tries to figure it out)

    Returns:
        merged dataframe
    """
    if on is None:
        # try to find common columns that make sense for joining
        commonCols = set(outageDf.columns) & set(weatherDf.columns)
        potentialKeys = ['fips_code', 'date', 'timestamp', 'county']
        on = [c for c in potentialKeys if c in commonCols]

        if not on:
            raise ValueError("couldnt find common columns to merge on, specify them manually")

    merged = pd.merge(outageDf, weatherDf, on=on, how='left')

    print(f"merged data: {len(merged)} records")
    print(f"records with missing weather: {merged['weather_code'].isna().sum()}")

    return merged


def calculateOutageDuration(df: pd.DataFrame, startCol: str = 'start_time',
                           endCol: str = 'end_time') -> pd.DataFrame:
    """
    calculates outage duration in minutes (our target variable)

    Args:
        df: dataframe with start and end time columns
        startCol: name of start time column
        endCol: name of end time column

    Returns:
        dataframe with duration_minutes column added
    """
    df = df.copy()

    start = pd.to_datetime(df[startCol])
    end = pd.to_datetime(df[endCol])

    df['duration_minutes'] = (end - start).dt.total_seconds() / 60

    # drop any weird negative durations or nulls
    invalidCount = (df['duration_minutes'] <= 0).sum() + df['duration_minutes'].isna().sum()
    if invalidCount > 0:
        print(f"warning: found {invalidCount} invalid durations, these will be dropped")
        df = df[df['duration_minutes'] > 0].dropna(subset=['duration_minutes'])

    return df


def prepareFeatures(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    prepares the final feature matrix and target variable

    selects just the columns we need for the model based on the PRD:
    - temporal: year, month, day, hour, dayofweek
    - spatial: fips_code, state_code
    - weather: weather_code

    Args:
        df: the preprocessed dataframe

    Returns:
        tuple of (X features dataframe, y target series)
    """
    featureCols = ['year', 'month', 'day', 'hour', 'dayofweek',
                   'fips_code', 'state_code', 'weather_code']

    # check which features we actually have
    availableFeatures = [c for c in featureCols if c in df.columns]
    missingFeatures = [c for c in featureCols if c not in df.columns]

    if missingFeatures:
        print(f"warning: missing features: {missingFeatures}")

    X = df[availableFeatures].copy()
    y = df['duration_minutes'].copy()

    return X, y


def runFullPipeline(outageDf: pd.DataFrame, weatherDf: pd.DataFrame,
                    timestampCol: str = 'start_time') -> Tuple[pd.DataFrame, pd.Series]:
    """
    runs the complete preprocessing pipeline end to end

    this is the main function you probably want to call - it handles
    all the steps in order and returns model-ready data

    Args:
        outageDf: raw EAGLE-I data
        weatherDf: raw NOAA data
        timestampCol: which column has the outage start time

    Returns:
        tuple of (X, y) ready for model training
    """
    # extract temporal features
    df = extractTemporalFeatures(outageDf, timestampCol)

    # calculate duration
    df = calculateOutageDuration(df)

    # merge weather
    df = mergeWeatherData(df, weatherDf)

    # prep final features
    X, y = prepareFeatures(df)

    print(f"\nfinal dataset: {len(X)} samples, {len(X.columns)} features")

    return X, y
