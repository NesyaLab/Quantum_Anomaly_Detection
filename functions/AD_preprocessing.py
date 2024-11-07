"Detection visualization functions"




import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from statistics import mean
import random
import gdown
import itertools




def generate_dataset(normal_sample_type: str, normal_sample_params: dict,
                     outlier_sample_type: str, outlier_sample_params: dict) -> tuple:
    """
    Generates a dataset containing both normal and outlier samples based on specified distributions 
    and parameters. Normal and outlier samples are shuffled to randomize order, and timestamps are 
    assigned sequentially.

    Args:
        normal_sample_type (str): The distribution type for normal samples. Options are 'uniform', 'normal', 
                                  'exponential', or 'poisson'.
        normal_sample_params (dict): Parameters for the normal sample distribution, passed as keyword arguments 
                                     to the corresponding numpy random function.
        outlier_sample_type (str): The distribution type for outlier samples. Options are 'uniform', 'normal', 
                                   'exponential', or 'poisson'.
        outlier_sample_params (dict): Parameters for the outlier sample distribution, passed as keyword arguments 
                                      to the corresponding numpy random function.

    Returns:
        dataset (list of tuples): A list of tuples, where each tuple contains a timestamp and a value. 
                                  Timestamps are generated sequentially.
        outlier_indices (set): A set of indices representing the positions of the outlier samples in the dataset.

    Raises:
        ValueError: If either the `normal_sample_type` or `outlier_sample_type` is unsupported.
    """
    if normal_sample_type == 'uniform':
        normal_series = np.random.uniform(**normal_sample_params)
    elif normal_sample_type == 'normal':
        normal_series = np.random.normal(**normal_sample_params)
    elif normal_sample_type == 'exponential':
        normal_series = np.random.exponential(**normal_sample_params)
    elif normal_sample_type == 'poisson':
        normal_series = np.random.poisson(**normal_sample_params)
    else:
        raise ValueError("Unsupported normal sample type")

    if outlier_sample_type == 'uniform':
        outlier_series = np.random.uniform(**outlier_sample_params)
    elif outlier_sample_type == 'normal':
        outlier_series = np.random.normal(**outlier_sample_params)
    elif outlier_sample_type == 'exponential':
        outlier_series = np.random.exponential(**outlier_sample_params)
    elif outlier_sample_type == 'poisson':
        outlier_series = np.random.poisson(**outlier_sample_params)
    else:
        raise ValueError("Unsupported outlier sample type")

    values = np.concatenate((normal_series, outlier_series))

    indices = np.arange(len(values))

    np.random.shuffle(indices)
    values = values[indices]

    num_points = len(values)
    times = [i for i in range(num_points)]

    dataset = [(int(i), j) for i, j in zip(times, values)]

    outlier_indices = set(np.where(indices >= len(normal_series))[0])

    return dataset, outlier_indices




def scale_dataset(dataset, new_min=1, new_max=10):
    """
    Scales the values in the dataset to a specified range [new_min, new_max], maintaining the relative 
    proportions of the original values.

    Args:
        dataset (list of tuples): A list of (timestamp, value) pairs representing the dataset (Time_series).
        new_min (float, optional): The minimum value of the scaled range. Default is 1.
        new_max (float, optional): The maximum value of the scaled range. Default is 10.

    Returns:
        scaled_dataset (list of tuples): A list of (timestamp, scaled_value) pairs, where each value 
                                         has been scaled to the specified range.
    """
    all_values = [value for _, value in dataset]

    original_min = min(all_values)
    original_max = max(all_values)

    def scale_value(value):
        return new_min + (value - original_min) * (new_max - new_min) / (original_max - original_min)

    scaled_dataset = [(timestamp, scale_value(value)) for timestamp, value in dataset]

    return scaled_dataset




def load_dataset_from_csv(file_path: str, time_column: str, value_column: str) -> tuple:
    """
    Loads a dataset from a CSV file, mapping timestamps to values and returning the dataset along 
    with the original time values (so they can be used/displayed if needed).

    Args:
        file_path (str): The path to the CSV file.
        time_column (str): The name of the column containing time data.
        value_column (str): The name of the column containing value data.

    Returns:
        dataset (list of tuples): A list of (timestamp, value) pairs with sequentially generated timestamps.
        original_times (np.ndarray): An array of original time values from the CSV file.
    """
    df = pd.read_csv(file_path)

    values = df[value_column].values

    original_times = df[time_column].values

    num_points = len(values)
    times = list(range(num_points))

    dataset = [(int(i), j) for i, j in zip(times, values)]

    return dataset, original_times




def load_partial_dataset_from_csv(file_path: str, time_column: str, value_column: str, start: int, end: int) -> tuple:
    """
    Loads a portion of the dataset from a CSV file and renumbers timestamps from 0 to (end - start).

    Args:
        file_path (str): Path to the CSV file.
        time_column (str): Name of the column containing timestamp data.
        value_column (str): Name of the column containing value data.
        start (int): Starting index of the data range to load.
        end (int): Ending index of the data range to load.

    Returns:
        dataset (list of tuples): A list of (timestamp, value) pairs with renumbered timestamps.
        original_times (np.ndarray): An array of original time values from the selected range.
    """
    df = pd.read_csv(file_path)

    values = df[value_column].values[start:end]
    original_times = df[time_column].values[start:end]

    num_points = len(values)
    times = list(range(num_points))  

    dataset = [(int(i), j) for i, j in zip(times, values)]

    return dataset, original_times




def split_dataset_with_best_batch_size(dataset, overlap=2, batch_sizes=[7, 8, 9, 10]):
    """
    Based on available batch sizes and the desired overlap between the batches, tests the dataset in order to split it in the most balanced way,
    then proceeds to actually effect the split.

    Args:
        dataset (list of tuples): The dataset to split, represented as a list of (timestamp, value) pairs.
        overlap (int, optional): The number of overlapping samples between consecutive batches. Default is 2.
        batch_sizes (list of int, optional): List of possible batch sizes to test. Default is [7, 8, 9, 10].

    Returns:
        best_batches (list of lists): A list of batches, where each batch is a list of (timestamp, value) pairs.
        best_batch_size (int): The batch size that results in the largest final batch.
    """
    num_samples = len(dataset)
    best_batch_size = None
    best_batches = []
    max_last_batch_size = 0

    for batch_size in batch_sizes:
        batches = []
        start = 0

        while start < num_samples:
            batch = dataset[start:start + batch_size]
            batches.append(batch)
            start += (batch_size - overlap)

        last_batch_size = len(batches[-1])

        if last_batch_size > max_last_batch_size:
            max_last_batch_size = last_batch_size
            best_batch_size = batch_size
            best_batches = batches

    return best_batches, best_batch_size
