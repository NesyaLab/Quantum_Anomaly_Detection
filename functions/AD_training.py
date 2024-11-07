"Training for alpha and beta functions"




import numpy as np
import math
from typing import Tuple, List, Dict
from statistics import mean
import random
import itertools




def grid_search_alpha_beta(X, alpha_range=np.linspace(-1, 0, 10)):
    """
    First grid-search approach for single batch (or small dataset) application and testing. 
    Performs a grid search over a range of alpha values to find the optimal (alpha, beta) 
    pair for minimizing the QAOA cost function on a given dataset.

    Args:
        X (list of tuples): The dataset represented as a list of (timestamp, value) pairs.
        alpha_range (np.ndarray, optional): A range of alpha values to search. Default is a linearly spaced 
                                            range from -1 to 0 with 10 values.

    Returns:
        best_alpha (float): The alpha value that results in the minimum cost.
        best_beta (float): The corresponding beta value (1 + alpha) for the optimal alpha.
        best_cost (float): The minimum cost value achieved.
        best_state (np.ndarray): The QAOA state that achieves the best cost.
    """
    best_alpha = None
    best_beta = None
    best_cost = float('inf')
    best_state = None

    print(f"Dataset:\n{X}\n")

    for alpha in alpha_range:
        beta = 1 + alpha

        ad_qaoa = AD_QAOA(X, alpha=alpha, beta=beta)

        states, _ = ad_qaoa.solve_qubo()
        qaoa_state = np.array(states[0])

        M = ad_qaoa.matrix_M()

        df = pd.DataFrame(M)
        df_rounded = df.round(3) 
        print(f"Matrix M for Alpha = {alpha} and Beta = {beta}:\n{df_rounded}\n")

        qaoa_cost = ad_qaoa.cost_function(M, qaoa_state)

        best_classical_state, best_classical_cost = ad_qaoa.find_min_cost(M)

        print(f"Alpha: {alpha}, Beta: {beta}")
        print(f"QAOA State: {qaoa_state}, QAOA Cost: {qaoa_cost}")
        num_ones = np.sum(qaoa_state)
        total_length = len(qaoa_state)
        percentage_ones = (num_ones / total_length)
        print(f"Percentage of '1' in QAOA state: {percentage_ones}")
        print(f"Approximation ratio: {qaoa_cost / best_classical_cost}")
        print(f"Classical Best State: {best_classical_state}, Classical Max Cost: {best_classical_cost}\n")

        if qaoa_cost < best_cost:
            best_cost = qaoa_cost
            best_alpha = alpha
            best_beta = beta
            best_state = qaoa_state

    return best_alpha, best_beta, best_cost, best_state




def rank_grid_search(X, alpha_range=np.linspace(-1, 0, 10)):
    """
    Performs a grid search over a range of alpha values, ranks the configurations by the number of selected 
    elements as centers for the coverage (string rank), and returns a sorted list of results.

    Args:
        X (list of tuples): The dataset represented as a list of (timestamp, value) pairs as a usual time series.
        alpha_range (np.ndarray, optional): A range of alpha values to search. Default is a linearly spaced 
                                            range from -1 to 0 with 10 values.

    Returns:
        sorted_results (list of dicts): A list of dictionaries containing details for each configuration, 
                                        sorted by 'string_rank'. Each dictionary includes:
                                        - 'alpha': The alpha value used.
                                        - 'beta': The corresponding beta value (1 + alpha).
                                        - 'qaoa_state': The QAOA solution state.
                                        - 'qaoa_cost': The cost for the QAOA state.
                                        - 'classical_cost': The minimum cost obtained classically.
                                        - 'approx_ratio': The approximation ratio (QAOA cost / classical cost).
                                        - 'string_rank': The count of '1's in the QAOA state.

    """
    results = []

    for alpha in alpha_range:
        beta = 1 + alpha

        ad_qaoa = AD_QAOA(X, alpha=alpha, beta=beta)

        states, _ = ad_qaoa.solve_qubo()
        qaoa_state = np.array(states[0])

        M = ad_qaoa.matrix_M()
        df = pd.DataFrame(M).round(3)

        qaoa_cost = ad_qaoa.cost_function(M, qaoa_state)
        best_classical_state, best_classical_cost = ad_qaoa.find_min_cost(M)

        if best_classical_cost != 0:
            approximation_ratio = qaoa_cost / best_classical_cost
        else:
            approximation_ratio = 1

        string_rank = np.sum(qaoa_state)

        if 0 < string_rank < len(qaoa_state):
            results.append({
                'alpha': alpha,
                'beta': beta,
                'qaoa_state': qaoa_state,
                'qaoa_cost': qaoa_cost,
                'classical_cost': best_classical_cost,
                'approx_ratio': approximation_ratio,
                'string_rank': string_rank
            })

    sorted_results = sorted(results, key=lambda x: x['string_rank'], reverse=True)

    return sorted_results




def collect_normalized_rank_data(all_epoch_results):
    """
    Collects and normalizes the rank data from multiple epochs of QAOA results.

    Args:
        all_epoch_results (list of lists): A list where each element is a list of dictionaries representing 
                                           results for an epoch. Each dictionary contains 'alpha', 'beta', 
                                           'qaoa_state', and 'string_rank'.

    Returns:
        alpha_values (list of float): A list of alpha values from all epochs.
        beta_values (list of float): A list of beta values corresponding to each alpha.
        normalized_rank_values (list of float): A list of normalized rank values, calculated as the ratio 
                                                of 'string_rank' to the length of 'qaoa_state' for each result.
    """
    alpha_values = []
    beta_values = []
    normalized_rank_values = []

    for epoch_results in all_epoch_results:
        for result in epoch_results:
            string_length = len(result["qaoa_state"])
            normalized_rank = result["string_rank"] / string_length
            alpha_values.append(result["alpha"])
            beta_values.append(result["beta"])
            normalized_rank_values.append(normalized_rank)

    return alpha_values, beta_values, normalized_rank_values




def calculate_mean_alpha_beta(all_epoch_results, selected_position):
    """
    Calculates the mean alpha and beta values based on a specific position in the ranking.

    Args:
        all_epoch_results (list of lists): A list where each element is a list of dictionaries representing 
                                           results for an epoch. Each dictionary should contain 'alpha' and 'beta'.
        selected_position (int): The ranking position to select for averaging alpha and beta values.

    Returns:
        alpha_mean (float): The mean alpha value for the specified ranking position across epochs.
        beta_mean (float): The mean beta value for the specified ranking position across epochs.
        If no results are found at the specified position, returns (None, None).
    """
    selected_alpha_beta = []

    for epoch_results in all_epoch_results:
        if selected_position - 1 < len(epoch_results):
            result = epoch_results[selected_position - 1]
            selected_alpha_beta.append((result["alpha"], result["beta"]))

    if selected_alpha_beta:
        alpha_mean = np.mean([alpha for alpha, _ in selected_alpha_beta])
        beta_mean = np.mean([beta for _, beta in selected_alpha_beta])
        return alpha_mean, beta_mean
    else:
        print(f"No configurations found at position {selected_position}.")
        return None, None
