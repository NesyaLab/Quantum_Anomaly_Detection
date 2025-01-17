# Quantum Anomaly Detection
This repository contains the implementation of an hybrid Quantum-Classical Anomaly Detection algorithm, which leverages quantum computing techniques to detect anomalies in time series data. The project is built around Quantum Approximate Optimization Algorithm (QAOA) and involves generating datasets, constructing quantum circuits, and optimizing parameters to identify anomalies.

## What's in here?
In this repository you can find:
- `class`: Contains the AD_QAOA class implementation, which serves as the core of the Quantum Anomaly Detection algorithm. It handles the construction of QUBO matrices, execution of the QAOA, and integration of classical anomaly detection methodologies.
- `functions`: A collection of utility modules, each addressing a specific aspect of the anomaly detection pipeline:
  - `AD_utilities.py`: General-purpose utilities for visualization, processing batch results, and executing QAOA on multiple datasets.
  - `AD_preprocessing.py`: Functions for preparing and scaling datasets, as well as splitting time series data into overlapping batches.
  - `AD_training.py`: Methods for grid search optimization, ranking results, and parameter tuning for QAOA.
  - `AD_detection.py`: Core anomaly detection algorithms leveraging QAOA and coverage-based methods. 
- `documentation`: Contains detailed explanations of the project, including theoretical background, methodology, and implementation details. Also includes user guides and API references for the repository.
- `data`: A directory for storing datasets used for testing and evaluation. Includes both synthetic datasets generated during execution and real-world datasets downloaded or processed for anomaly detection.
- `execute.ipynb`: A Jupyter Notebook providing a step-by-step execution of the Quantum Anomaly Detection pipeline, from data preparation to model execution, visualization, and anomaly detection.
- `benchmark.ipynb`: A Jupyter Notebook dedicated to comparing the Quantum Anomaly Detection algorithm with classical anomaly detection methods (e.g., DBSCAN, LOF, One-Class SVM). Includes performance metrics, visualizations, and insights.

## Use this repository
If you want to use the code in this repository in your projects, please cite explicitely our work, and
- Clone this repository with `git clone https://github.com/NesyaLab/Quantum_Anomaly_Detection`.
- Install the requirements with `pip install -r requirements.txt`.
  
Before running any experiments, ensure that the `config.txt` file is correctly set up.

## License
This project is licensed under the MIT License.

## Contributing
We welcome contributions to enhance the functionality and performance of the models. Please submit pull requests or open issues for any improvements or bug fixes.

