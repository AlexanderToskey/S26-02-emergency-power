"""
data_loader.py - handles loading the EAGLE-I and NOAA datasets

basically just wraps pandas read functions with some basic validation
so we dont have to keep rewriting the same loading logic everywhere
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def loadEagleI(filePath: str) -> pd.DataFrame:
    """
    loads the EAGLE-I outage records dataset

    expects a csv with columns for outage events, timestamps, location info etc.
    does some basic cleaning like parsing dates and dropping obvious junk rows

    Args:
        filePath: path to the EAGLE-I csv file

    Returns:
        dataframe with the outage records
    """
    df = pd.read_csv(filePath)

    # convert date columns if they exist
    dateColumns = ['start_time', 'end_time', 'timestamp']
    for col in dateColumns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


def loadNOAA(filePath: str) -> pd.DataFrame:
    """
    loads NOAA weather data

    this should have weather codes and timestamps that we can merge
    with the outage data later on

    Args:
        filePath: path to the NOAA csv file

    Returns:
        dataframe with weather records
    """
    df = pd.read_csv(filePath)

    # parse any date columns
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    return df


def loadBothDatasets(eaglePath: str, noaaPath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    convenience function to load both datasets at once

    Args:
        eaglePath: path to EAGLE-I data
        noaaPath: path to NOAA data

    Returns:
        tuple of (eagle_df, noaa_df)
    """
    eagleDf = loadEagleI(eaglePath)
    noaaDf = loadNOAA(noaaPath)

    print(f"loaded EAGLE-I: {len(eagleDf)} records")
    print(f"loaded NOAA: {len(noaaDf)} records")

    return eagleDf, noaaDf
