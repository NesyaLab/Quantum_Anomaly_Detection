# Quantum Anomaly Detection

## Overview

This repository contains the implementation of a Quantum Anomaly Detection algorithm, which leverages quantum computing techniques to detect anomalies in time series data. The project is built around Quantum Approximate Optimization Algorithm (QAOA) and involves generating datasets, constructing quantum circuits, and optimizing parameters to identify anomalies. This project is carried out in close collaboration with the Nesya Laboratory at the Department of Engineering (DIET) of Sapienza University of Rome. Special thanks to Massimo Panella and Leonardo Lavagna for their valuable contributions and support.

## Repository Structure

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

## Prerequisites

Make sure you have Python installed on your machine.

Clone this repository to your local machine using:
```bash
git clone https://github.com/yourusername/quantum-anomaly-detection.git
cd quantum-anomaly-detection
```
Before running any experiments, ensure that the config.txt file is correctly set up (like folder function dependencies).



Special thanks to the contributors and the open-source community for their valuable tools and libraries that make quantum computing research accessible.
