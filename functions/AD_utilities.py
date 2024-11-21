"Anomaly detection Utilites functions"




import numpy as np
import pandas as pd
import math
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from statistics import mean
import random
import gdown
import time
import itertools
from AD_QAOA import AD_QAOA 
import functions.AD_preprocessing as preprocessing
import functions.AD_detection as detection
import functions.AD_training as training



def plot_training_time_series(dataset_train):
    """
    Plots the training time series.

    Args:
        dataset_train (list of tuples): The training dataset represented as a list of (timestamp, value) pairs.

    """
    times = np.array([t for t, _ in dataset_train])
    values = np.array([v for _, v in dataset_train])

    plt.figure(figsize=(10, 6))
    plt.plot(times, values, marker='o', color='goldenrod', linestyle='-', label='Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Training Time Series')
    plt.show()




def plot_test_time_series(dataset_test):
    """
    Plots the test time series.

    Args:
        dataset_test (list of tuples): The test dataset represented as a list of (timestamp, value) pairs.

    """
    times = np.array([t for t, _ in dataset_test])
    values = np.array([v for _, v in dataset_test])

    plt.figure(figsize=(10, 6))
    plt.plot(times, values, marker='o', color='blue', linestyle='-', label='Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Test Time Series')
    plt.show()




def plot_training_time_series_batches(dataset_train, overlap=2, batch_sizes=[7, 8, 9]):
    """
    Plots the training time series divided into batches, with vertical lines separating each batch.

    Args:
        dataset_train (list of tuples): The training dataset represented as a list of (timestamp, value) pairs.
        overlap (int, optional): The number of overlapping samples between consecutive batches. Default is 2.
        batch_sizes (list of int, optional): List of possible batch sizes to test for splitting the dataset. 
                                             Default is [7, 8, 9].
    """
    # print("Inside plot_training_time_series_batches:") # Debug
    # print("split_dataset_with_best_batch_size:", split_dataset_with_best_batch_size)  # Debug

    batches, best_batch_size = preprocessing.split_dataset_with_best_batch_size(dataset_train, overlap, batch_sizes)

    plt.figure(figsize=(10, 6))

    for batch in batches:
        batch_times = [t for t, _ in batch]
        batch_values = [v for _, v in batch]
        plt.plot(batch_times, batch_values, marker='o', linestyle='-', color='goldenrod')

    for i in range(len(batches) - 1):
        last_time_in_batch = batches[i][-1][0]
        plt.axvline(x=last_time_in_batch, color='darkslategray', linestyle='--', alpha=0.8)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Training Time Series Divided into Batches')
    plt.show()





def plot_benchmark_results(data, anomalies, title="Classical Model Anomaly Detection"):
    """
    Plots the dataset with anomalies highlighted for benchmarking a classical model.

    Args:
        times (list): A list of timestamps for the dataset.
        values (list): A list of values corresponding to each timestamp in the dataset.
        anomalies (list): A list of timestamps identified as anomalies.
        title (str, optional): The title of the plot. Default is "Classical Model Anomaly Detection".

    """
    times = np.array([t for t, _ in data])
    values = np.array([v for _, v in data])

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(times, values, 'bo-', label='Dataset')

    ax.plot(anomalies, values[np.isin(times, anomalies)], 'ro', label='Anomalies')

    ax.set_title(title)
    ax.set_xlabel("Timestamps")
    ax.set_ylabel("Values")
    ax.legend()
    plt.grid(True)
    plt.show()




def execute_batch_processing(batches, alpha_range=np.linspace(-1, 0, 10), selected_position=1):
    """
    Executes a grid search over alpha values for each batch, calculates the mean alpha and beta values,
    and collects normalized rank data.

    Args:
        batches (list of lists): A list where each element is a batch represented as a list of (timestamp, value) pairs.
        alpha_range (np.ndarray, optional): A range of alpha values to search for each batch. Default is 
                                            a linearly spaced range from -1 to 0 with 10 values.
        selected_position (int, optional): The ranking position used to calculate the mean alpha and beta values. 
                                           Default is 1.

    Returns:
        alpha_mean (float): The mean alpha value for the specified ranking position across batches.
        beta_mean (float): The mean beta value for the specified ranking position across batches.
        alpha_values (list of float): A list of alpha values from all batches.
        beta_values (list of float): A list of beta values corresponding to each alpha.
        normalized_rank_values (list of float): A list of normalized rank values for each batch.
    """
    all_batch_results = []

    start_total_time = time.time()

    for i, batch in enumerate(batches, 1):
        print(f"\nProcessing batch {i}/{len(batches)}...")

        start_batch_time = time.time()

        batch_results = training.rank_grid_search(batch, alpha_range=alpha_range)
        
        if not batch_results:
            print(f"Warning: rank_grid_search did not produce results for batch {i}. Skipping this batch.")
            all_batch_results.append([])  
            continue  
        
        print(f"Batch {i} results: {batch_results[0]}")
        all_batch_results.append(batch_results)

        end_batch_time = time.time()
        batch_time = end_batch_time - start_batch_time

        print(f"Batch {i} completed in {batch_time:.2f} seconds.")

    end_total_time = time.time()
    total_time = end_total_time - start_total_time

    valid_results = [result for result in all_batch_results if result]
    
    if valid_results:
        alpha_mean, beta_mean = training.calculate_mean_alpha_beta(valid_results, selected_position)
        print(f"\nMean Alpha: {alpha_mean}, Mean Beta: {beta_mean}")
    else:
        print("No valid results were found across all batches.")
        return None, None, [], [], []

    print(f"Process completed in {total_time:.2f} seconds.")

    alpha_values, beta_values, normalized_rank_values = training.collect_normalized_rank_data(valid_results)

    return alpha_mean, beta_mean, alpha_values, beta_values, normalized_rank_values





def execute_qaoa_on_batches(batches, model_name='cubic', model_params={}, alpha_mean=None, beta_mean=None):
    """
    Executes the QAOA class on each batch, associates centers with radii, and collects unique centers across all batches.

    Args:
        batches (list of lists): A list where each element is a batch represented as a list of (timestamp, value) pairs.
        model_name (str, optional): The name of the model used in the AD_QAOA class. Default is 'cubic'.
        model_params (dict, optional): Parameters for the model used in the AD_QAOA class. Default is an empty dictionary.
        alpha_mean (float, optional): The mean alpha value used to initialize AD_QAOA. If None, it must be calculated beforehand.
        beta_mean (float, optional): The mean beta value used to initialize AD_QAOA. If None, it must be calculated beforehand.

    Returns:
        unique_centers_with_radii (list of tuples): A list of unique centers with associated radii across all batches.
    """
    print("\nInitializing the final run...")
    
    all_centers_with_radii = []

    for i, batch in enumerate(batches, 1):
        print(f"\nExecuting QAOA model on batch {i}/{len(batches)}...")

        ad_qaoa = AD_QAOA(X=batch, model_name=model_name, model_params=model_params, radius_adjustment=True, 
                          alpha=alpha_mean, beta=beta_mean)

        top_n_states, qaoa_solution = ad_qaoa.solve_qubo()
        print(f"QAOA solution for batch {i}: {top_n_states}")

        M = ad_qaoa.matrix_M()
        df = pd.DataFrame(M).round(3)

        qaoa_cost = ad_qaoa.cost_function(M, np.array(top_n_states[0]))
        best_state, classical_sol = ad_qaoa.find_min_cost(M)
        approximation_ratio = qaoa_cost / classical_sol
        print(f"Batch {i} - QAOA cost: {qaoa_cost}, Classical optimal solution: {classical_sol}, Approximation ratio: {approximation_ratio}")

        # (Optional) Plot various components
        # ad_qaoa.plot_time_series()
        # ad_qaoa.plot_model()
        # ad_qaoa.visualize_anomalies()

        batch_centers_with_radii = ad_qaoa.associate_centers_with_radius()

        all_centers_with_radii.extend(batch_centers_with_radii)
        print(f"Batch {i} completed. Saved centers with radii: {batch_centers_with_radii}")

    unique_centers_with_radii = []
    seen_timestamps = set()

    for center_with_radius in all_centers_with_radii:
        timestamp = center_with_radius[0][0]
        if timestamp not in seen_timestamps:
            unique_centers_with_radii.append(center_with_radius)
            seen_timestamps.add(timestamp)

    print("\nResulting Set Coverage:")
    for i, centers_radii in enumerate(unique_centers_with_radii):
        print(f"Center {i}: {centers_radii}")

    return unique_centers_with_radii
