# cluster_maker
`cluster_maker` is a Python package for data preprocessing, clustering analysis, evaluation, and visualisation. It provides tools for applying clustering algorithms to datasets, comparing clustering results, and exporting analysed data.

## Allowed libraries
The package only uses:

- Python standard library  
- NumPy  
- pandas  
- matplotlib  
- SciPy  
- scikit-learn  

No other third-party libraries are required and/or are permitted.

## Main features
- Data preprocessing, including feature selection and standardisation.
- K-means clustering using both a manual implementation and scikit-learn.
- Agglomerative hierarchical clustering using different linkage methods.
- Evaluation of clustering results using inertia and silhouette scores.
- Elbow method analysis to help select an appropriate number of clusters.
- Visualisation of clustered datasets and evaluation results.
- Generation, analysis, and export of data using pandas and NumPy.


## Package root directory structure

- `cluster_maker/` - Main package directory.
  - `algorithms.py` - K-means clustering algorithms.
  - `agglomerative.py` - Agglomerative hierarchical clustering.
  - `preprocessing.py` - Feature selection and standardisation.
  - `evaluation.py` - Clustering evaluation metrics.
  - `plotting_clustered.py` - Clustering and elbow plots.
  - `interface.py` - High-level clustering interface.
  - `dataframe_builder.py` - Data generation and DataFrame creation.
  - `data_analyser.py` - Descriptive statistics and correlation analysis.
  - `data_exporter.py` - Data export functions.
- `demo/` - Example scripts 
- `data/` - CSV datasets used by the example scripts.
- `tests/` - Unit tests using the standard library `unittest`.

## Installation (local use)
From the root directory of the project (the directory containing `pyproject.toml`), run:

```bash
python -m pip install -e .
```

This installs the package in editable mode, making `cluster_maker` available for import. Changes to the Python source files take effect without reinstalling the package, so you can modify the code and re-run tests or demos directly.

Make sure you have activated the intended Python environment before installing the package.

## Notes on pyproject.toml and the *.egg-info directory
This project includes a small file named pyproject.toml.
You do not need to open or edit it. Its only purpose is to tell Python/pip
that this folder is a valid installable package. Without it, the command
`python -m pip install -e .` would fail.

When you run the installation command, pip automatically creates a directory
called something like:

`cluster_maker.egg-info/`

This folder contains package metadata used internally by Python (file lists,
version information, etc.). It is generated automatically and should not be
edited.
