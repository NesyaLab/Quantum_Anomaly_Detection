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

