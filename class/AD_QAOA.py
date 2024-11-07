"Quantum Anomaly Detection"

from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, NELDER_MEAD, ADAM
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer, GurobiOptimizer, MinimumEigenOptimizer
import numpy as np
import math
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from statistics import mean
import random
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
import math
from scipy.optimize import curve_fit
import itertools



class AD_QAOA:
    """
    Anomaly detection class using QAOA.
    """
    def __init__(self,
                 X: List[Tuple[int, float]],
                 alpha: float = -0.5,
                 beta: float = 0.5,
                 radius: float = 1,
                 top_n_samples: int = 5,
                 num_iterations: int = 10,
                 model_name: str = 'quadratic',
                 model_params: dict = {},
                 radius_adjustment: bool = False,
                 num_layers: int = 1,
                 debug=False):
        """
        Initializes the AD_QAOA class.
        Disclaimer: mixer selection work in progress.

        Args:
            X (List[Tuple[int, float]]): Time series data.
            alpha (float): Weight for the linear terms in the QUBO problem.
            beta (float): Weight for the quadratic terms in the QUBO problem.
            radius (float): Radius for the covering boxes.
            top_n_samples (int): Number of top samples to consider.
            num_iterations (int): Number of iterations for the COBYLA optimizer.
            model_name (str): Model selected for the detection pipeline.
            model_params (str): Model's parameters (if any).
            radius_adjustment (bool):  Enables the radius adjustment mechanism for the set covering.
            num_layers (int): Number of layers (p) to use in QAOA.
            debug (bool): Enables some debug prints throught the code.
        """
        self.X = X
        self.alpha = alpha
        self.beta = beta
        self.radius = radius
        self.top_n_samples = top_n_samples
        self.num_iterations = num_iterations
        self.model_name = model_name
        self.model_params = model_params
        self.radius_adjustment = radius_adjustment
        self.num_layers = num_layers  
        self.debug = debug




    def matrix_M(self) -> np.ndarray:
        """
        Builds the matrix M for the QUBO Anomaly Detection objective function. 
        """
        n = len(self.X)
        L = self.diag_M(self.X)
        Q = self.off_diag_M(self.X)
        M = np.zeros((n, n))

        for i in range(n):
            M[i, i] = L[i]
            for j in range(i + 1, n):
                M[i, j] = Q[i, j]
                M[j, i] = Q[i, j]

        if self.debug:
          print("(debug) L:\n", L)
          print("(debug) Q:\n", Q)
          print("(debug) M:\n", M)

        return M



    
    def off_diag_M(self, data: List[Tuple[int, float]]) -> np.ndarray:
        """
        Builds the off-diagonal terms (Q, quadratic contribution for the QUBO problem) for the corresponding M matrix.

        Args:
         data (List[Tuple[int, float]]): Time Series data.
        
        Returns:
         Q (np.ndarray): The symmetric matrix Q of the quadratic terms for the QUBO problem.
        """
        n = len(data)
        Q = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(np.array(data[i]), np.array(data[j]))
                Q[i, j] = d
                Q[j, i] = d

        return Q




    def compute_delta(self, data: np.ndarray, model_values: np.ndarray) -> List[float]:
        """
        Creates the diagonal terms (delta, linear contribution for the QUBO problem) for the corresponding M matrix,
        computing the difference between the data sample and the corresponding model fitting values.

        Args:
         data (np.ndarray): Time Series data.
         model_values (np.ndarray): Corresponding model values for the fitting.
         
        Returns:
         List[float]: List of absolute differences data sample/model sample.
        """
        return [abs(sample_value - model_value.item()) for sample_value, model_value in zip(data, model_values)]



        
    def inverse_transform(self, delta_values: List[float], scale_factor: float = 0.5) -> List[float]:
        """
        Applies an inverse transformation to delta values. Larger values ​​of delta become smaller, and vice versa.
        This allows the minimization problem for the QUBO formulation to correctly identify anomalies based on the highest values
        in the model fitting vector (high differences bewteen the model and the data sample). Also works as a normalization for the values.

        Args:
         delta_values (List[float]): List of absolute differences data sample/model sample.
         scale_factor (float): Scaling factor for the normalization and the transformation of the values.
         
        Returns:
         transfrmed_values (List[float]): List of transformed values to be used in the building of the diagonal of M for the QUBO problem.
        """
        transformed_values = []
        for delta in delta_values:
            if delta != 0:
                transformed_value = scale_factor / delta
            else:
                transformed_value = scale_factor  
            transformed_values.append(transformed_value)
        return transformed_values



    
    def diag_M(self, data: np.ndarray, scale_factor: float = 0.5) -> List[float]:  
        """
        Computes the diagonal of the matrix M, using the normalization and the inversion of the data values, having selected a proper fitting model.
        Adjustable scale factor: for a standard range of values use default; for very small dataset values use smaller factors.

        Args:
         data (np.ndarray): Time Series data.
         scale_factor (float): Scaling factor for the normalization and the transformation of the values.

        Returns:
         transformed_delta (List(float)): The diagonal component of the effective matrix M for the QUBO problem.

        Raises:
         ValueError: If a not supported model type is selected.
        """
        data_values = np.array([value for _, value in data])

        models = {
            'linear': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 1), x),
            'quadratic': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 2), x),
            'cubic': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 3), x),
            'sigmoid': lambda x: self.sigmoid(x, *curve_fit(self.sigmoid, x.flatten(), data_values, maxfev=2000)[0]),
            'moving_average': lambda x: self.moving_average_expanded(data_values, self.model_params.get('window_size', 2)),
        }

        model_name = self.model_name  

        if model_name in models:
            model_func = models[model_name]
            model_values = model_func(np.array([t for t, _ in data]).reshape(-1, 1))
            delta = self.compute_delta(data_values, model_values)

            transformed_delta = self.inverse_transform(delta, scale_factor=scale_factor)

            return transformed_delta
        else:
            raise ValueError(f"Model {model_name} is not supported.")


        
            
    def distance(self, point1: np.ndarray, point2: np.ndarray, kind: str = "absolute_difference") -> float:
        """
        Calculates the (absolute) distance between two points.
    
        Args:
            point1 (np.ndarray): First point.
            point2 (np.ndarray): Second point.
            kind (str): the kind of distance to be used. Default is "absolute".
    
        Returns:
            float: the distance between point1 and point2.
    
        Raise:
            NotImplementedError: If a not supported distance type is selected.
            ValueError: if at least one of the points is not in the bidimensional array form (time_series' data point).
        """
        

        if point1.shape != (2,) or point2.shape != (2,):
            raise ValueError("Both points have to be bidimensional arrays with 2 elements (x, y).")

        if kind == "euclidean":
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif kind == "chebyshev":
            return np.max(np.abs(point1 - point2))
        elif kind == "manhattan":
            return np.sum(np.abs(point1 - point2))
        elif kind == "absolute_difference":
            return np.abs(point1[1] - point2[1])
        else:
            raise NotImplementedError




    def moving_average_expanded(self, series: np.ndarray, window_size: int = 2) -> np.ndarray:
        """
        Calculates the moving average model (expanded to match the lenght of the list of samples) for the group of sample given
        a selected window size (default is 2).

        Args:
         series (np.ndarray): Data samples used.
         window_size (int): The shifting window size for the computation.

        Returns:
         expanded_moving_avg (np.ndarray): Vector of the moving average model (with position currespondance with the original data samples).
        """

        series = np.asarray(series)

        if window_size < 1:
            raise ValueError("Window size must be at least 1.")
        if window_size > len(series):
            raise ValueError("Window size can't be greater than the length of the dataset.")

        moving_avg = np.convolve(series, np.ones(window_size) / window_size, mode='valid')
        expanded_moving_avg = np.repeat(moving_avg, window_size)
        expanded_moving_avg = expanded_moving_avg[:len(series)]

        return expanded_moving_avg



    
    def plot_model(self):
        """
        Creates the plotting showcasing the model fitting on the Time Series.

        Raises:
         ValueError: If a non supported model is selected.
        """
        data_values = np.array([value for _, value in self.X])
        times = np.array([t for t, _ in self.X]).reshape(-1, 1)

        models = {
            'linear': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 1), x),
            'quadratic': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 2), x),
            'cubic': lambda x: np.polyval(np.polyfit(x.flatten(), data_values, 3), x),
            'moving_average': lambda x: self.moving_average_expanded(data_values, self.model_params.get('window_size', 2)),
        }

        if self.model_name in models:
            model_func = models[self.model_name]
            expected_values = model_func(times)

            plt.figure(figsize=(10, 6))
            plt.plot(times, data_values, 'bo-', label='Dataset')
            plt.plot(times, expected_values, 'r-', label=f'Model ({self.model_name})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Time series fitting')
            plt.legend()
            plt.show()
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")



            
    def plot_time_series(self):
        """
        Creates the plotting showcasing the Time Series.
        """
        times = np.array([t for t, _ in self.X])
        values = np.array([v for _, v in self.X])

        plt.figure(figsize=(10, 6))

        plt.plot(times, values, marker='o', color='blue', linestyle='-', label='Time Series')

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series')
        if self.debug:
            plt.legend()
        plt.show()




    def plot_distances_with_arrows(self):
        """
        Creates the plotting showcasing the distance between a selected sample (first_sample_value) and the rest of the Time Series (batch).
        """
        data_values = np.array([value for _, value in self.X])
        times = np.array([t for t, _ in self.X])

        first_sample_value = data_values[3] 
        first_sample_time = times[3]

        plt.figure(figsize=(8, 6))

        plt.scatter(times, data_values, color='blue', label='Data Points')

        for i in range(0, len(times)):
          plt.arrow(first_sample_time, first_sample_value,
                    times[i] - first_sample_time, data_values[i] - first_sample_value,
                      head_width=0.05, head_length=0.1, fc='red', ec='red', alpha=0.5)

        plt.scatter([first_sample_time], [first_sample_value], color='green', s=100, label='Example')

        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Distances')
        if self.debug:
            plt.legend()
        plt.show()




    def radius_adj(self, centers):
        """
        Radius adjustment algorithm. Determinates the best radius value for the centers and batch in exam.
        Default the radius is set to 1.00 and then is tried enlarging or reducing for the covering achievement.
        Adjust the exclusive tolerance (0.33 standard) for the max_non_center_value.
        Adjust the inclusive tolerance ( /2 standard) for the normal values inclusion.

        Args:
         centers: List of centers selected as 1s in the QAOA quantum state solution.

        Returns:
         self.radius (float): The optimal radius value identified for the batch at hand.
        """
        if self.radius_adjustment:
            X_v = [(i[0], i[1]) for i in self.X]
            not_centers = [i for i in range(len(self.X)) if i not in [c[0] for c in centers]]

            if not_centers:
                max_non_center_value = max([X_v[i][1] for i in not_centers])
                r = 0
                std_dev = np.std([i[1] for i in X_v])
                data_range = max([i[1] for i in X_v]) - min([i[1] for i in X_v])

                for i, center in enumerate(centers):
                    center_coords = np.array([center[0], center[1]])

                    for not_center_index in not_centers:
                        not_center_coords = np.array([X_v[not_center_index][0], X_v[not_center_index][1]])

                        distanza = np.linalg.norm(center_coords - not_center_coords)

                        if distanza < (0.33 * max_non_center_value):   
                            r = max(r, distanza)

                r += (std_dev / 2)  

                if r <= 1.0:
                    if data_range < 0.5:
                        print(f"The radius of 1.0 is too large for this dataset range.: {data_range}")
                        self.radius = 0.067
                        print("Radius:", r)
                        print("Radius adjusted check_ok")
                    else:
                        self.radius = 1.0
                        print("Radius:", r)
                        print("Radius adjusted check_ok")
                else:
                    self.radius = r
                    print("Radius:", self.radius)
                    print("Radius adjusted check_ok")

            elif not not_centers:
                self.radius = 1
                print("Radius:", self.radius)
                print("No adjustment")

        return self.radius



    
    def solve_qubo(self) -> Tuple[List[List[int]], dict]:
        """
        Solves the QUBO Anomaly Detection problem using the parameters defined in the class.
        
        Returns:
         top_n_states (List[List[int]]): The best binary solutions found for the QUBO problems.
         variables_dict (dict): A dictionary with the values of the variables.
        """

        M = self.matrix_M()

        linear_terms = self.alpha * (np.diag(M))
        quadratic_terms = {}

        for i in range(len(self.X)):
            for j in range(i + 1, len(self.X)):
                if M[i][j] != 0:
                    quadratic_terms[("x" + str(i), "x" + str(j))] = self.beta * M[i][j]

        qubo = QuadraticProgram()
        for i in range(len(self.X)):
            qubo.binary_var(name="x" + str(i))

        qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

        meo = MinimumEigenOptimizer(QAOA(sampler=Sampler(), reps = self.num_layers, optimizer=COBYLA(maxiter=self.num_iterations)))
        result = meo.solve(qubo)

        samples = []
        for sample in result.samples[:self.top_n_samples]:
            samples.append(sample)

        top_n_states = []
        for i in range(self.top_n_samples):
            top_n_states.append(list([int(x) for x in list(samples[i].x)]))

        return top_n_states, result.variables_dict




    def centers_storage(self, state=None):
        """
        Stores the center coordinates corresponding to 1s in the QAOA solution (default is the first state if none is provided).
    
        Args:
         state (List[int], optional): A binary list representing the QAOA solution state. If None, the function will 
                                      use the first quantum state (most counts).
        Returns:
         centers (List[Tuple[int, float]]): A list of tuples, where each tuple contains a timestamp and its corresponding 
                                            value for each selected center. These centers are identified by the positions 
                                            in the solution string where the value is '1'.
        """
        if state is None:
            _, best_solution = self.solve_qubo()
            non_zero_vars = [key for key, value in best_solution.items() if value > 0.5]
            time_stamps = [self.X[int(s.replace("x", ""))][0] for s in non_zero_vars]
            selected_values = [self.X[int(s.replace("x", ""))][1] for s in non_zero_vars]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]

            if self.debug:
                print('selected_centers:', centers)
            return centers
        else:
            best_solution = state
            non_zero_vars = [value for value in state if value > 0.5]
            time_stamps = [self.X[i][0] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            selected_values = [self.X[i][1] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]

            if self.debug:
                print('selected_centers:', centers)
            return centers



    
    def detect_anomalies(self, state=None):
        """
        Detects anomalies based on the selected centers in the QAOA solution state.
    
        Args:
         state (List[int], optional): A binary list representing the QAOA solution state. If None, the function will 
                                         use the first quantum state (most counts).
    
        Returns:
         boxes (List[Tuple[int, float]]): A list of tuples representing the coverage spheres for each center, 
                                             where each box is centered around selected points and adjusted to cover 
                                             surrounding data points.
        """
        if state is None:
            _, best_solution = self.solve_qubo()
            non_zero_vars = [key for key, value in best_solution.items() if value > 0.5]
            time_stamps = [self.X[int(s.replace("x", ""))][0] for s in non_zero_vars]
            selected_values = [self.X[int(s.replace("x", ""))][1] for s in non_zero_vars]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]
            
            if self.debug:
                print('selected_centers:', centers)
                
            boxes = self.covering_boxes(centers)
            return boxes
        else:
            best_solution = state
            non_zero_vars = [value for value in state if value > 0.5]
            time_stamps = [self.X[i][0] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            selected_values = [self.X[i][1] for i in range(len(state)) if non_zero_vars[i] > 0.5]
            centers = [(ts, val) for ts, val in zip(time_stamps, selected_values)]
            boxes = self.covering_boxes(centers)
            return boxes

        
        
    def associate_centers_with_radius(self, state=None):
        """
        Associates each center with the corresponding calculated radius.
    
        Args:
            state (list, optional): A binary list representing the QAOA solution state. If None, the function will 
                                    retrieve the best solution available.
    
        Returns:
            centers_with_radius (list of tuples): A list of (center, radius) pairs, where each center has an associated radius.
    
        """
        centers = self.centers_storage(state=state)
    
        radius = self.radius_adj(centers)
    
        if self.debug:
            print("Radius from radius_adj:", radius)
    
        centers_with_radius = [(center, radius) for center in centers]
        
        if self.debug:
            print("List of centers and radiuses:", centers_with_radius)
    
        return centers_with_radius




    def visualize_anomalies(self, state=None):
        """
        Visualizes anomalies detected by the QAOA model, highlighting centers and coverage areas on a scatter plot.
        Adjust plot title.
        
        Args:
         state (List[int], optional): A binary list representing the QAOA solution state. If None, the function will 
                                         use the first quantum state (most counts).
        """
        plt.figure(figsize=(10, 6))

        boxes = self.detect_anomalies(state)
        X_t = [i[0] for i in self.X]
        X_v = [i[1] for i in self.X]

        for box in boxes:
            plt.plot(box[0], box[1], color="orange")

        plt.scatter(X_t, X_v)

        centers = self.centers_storage(state)
        time_stamps = [center[0] for center in centers]
        selected_values = [center[1] for center in centers]

        if self.debug:
            print("selected_values:", centers)

        plt.scatter(time_stamps, selected_values, color="red")

        plt.axis('scaled')
        plt.title('Detection')    
        plt.show()



    
    def cost_function(self, M: np.ndarray, state: np.ndarray) -> float:
        """
        Calculates the cost for a given QAOA solution state based on the matrix M.
        
        Args:
         M (np.ndarray): A square symmetric matrix representing interactions between variables. The diagonal elements 
                         represent linear terms, while the off-diagonal elements represent quadratic terms.
         state (np.ndarray): A binary vector (1D array) representing the QAOA solution state. The length of `state` 
                             should match the dimensions of `M`.
    
        Returns:
         cost (float): The calculated cost for the given state, based on the weighted sum of linear and quadratic terms.
    
        Raises:
            ValueError: If the dimension of `state` does not match the dimension of `M`.
        """
        if len(state.shape) > 1:
            state = state[0]

        if len(state) != M.shape[0]:
            raise ValueError(f"State has dimension {len(state)} and instead should have dimension {M.shape[0]}.")

        M_off_diag = M - np.diag(np.diag(M))

        linear_terms = self.alpha * np.dot(np.diag(M), state)

        quadratic_terms = self.beta * np.dot(state.T, np.dot(M_off_diag, state))

        # cost = -(linear_terms + quadratic_terms)

        cost = linear_terms + quadratic_terms

        return cost



    
    def find_max_cost(self, M: np.ndarray) -> tuple:
        """
        Finds the state with the maximum cost for a given matrix M.
    
        Args:
            M (np.ndarray): A square symmetric matrix representing interactions between variables, 
                            with diagonal elements representing linear terms and off-diagonal elements 
                            representing quadratic terms.
    
        Returns:
            best_state (np.ndarray): A binary array representing the state that maximizes the cost function.
            max_cost (float): The maximum cost value obtained with the best state.
        """
        max_cost = float('-inf')
        best_state = None
    
        for state in itertools.product([0, 1], repeat=len(M)):
            state_array = np.array(state)
            current_cost = self.cost_function(M, state_array)
    
            if current_cost > max_cost:
                max_cost = current_cost
                best_state = state_array
    
        return best_state, max_cost
    


    
    def find_min_cost(self, M: np.ndarray) -> tuple:
        """
        Finds the state with the minimum cost for a given matrix M.
    
        Args:
            M (np.ndarray): A square symmetric matrix representing interactions between variables, 
                            with diagonal elements representing linear terms and off-diagonal elements 
                            representing quadratic terms.
    
        Returns:
            best_state (np.ndarray): A binary array representing the state that minimizes the cost function.
            min_cost (float): The minimum cost value obtained with the best state.
        """
        min_cost = float('inf')
        best_state = None
    
        for state in itertools.product([0, 1], repeat=len(M)):
            state_array = np.array(state)
            current_cost = self.cost_function(M, state_array)
    
            if current_cost < min_cost:
                min_cost = current_cost
                best_state = state_array
    
        return best_state, min_cost