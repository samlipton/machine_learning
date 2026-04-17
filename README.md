# Machine Learning 

This repository contains a collection of machine learning projects implemented using modular and reproducible workflows.

The analyses, provided as [Jupyter notebooks](https://github.com/samlipton/machine_learning/tree/master/notebooks) (`*.ipynb`), cover the following paradigms:
- **Supervised learning**: Support Vector Machines (SVM)
- **Ensemble learning**: Random Forests (RF)
- **Deep learning**: Neural Networks (NN)

## Philosophy

The repository is organized around the following objectives:
1. Apply multiple machine learning models to diverse datasets
2. Optimize data processing and model parametrization
3. Compare model performance across implementations and datasets
4. Build reproducible and modular pipelines

## Structure

The directory structure is intentionally lightweight and modular:
- **data/**: storage for raw, cleaned and processed datasets
- **scripts/**: reusable code snippets implementing core functionalities
- **projects/**: workflow-oriented modules organized around problem types
- **notebooks/**: exploratory analysis and methodological documentation

### Data

The datasets are downloaded into the **data/** directory from public repositories: 
- glass_id: [glass identification](https://archive.ics.uci.edu/dataset/42/glass+identification) based on oxide composition

### Scripts

The code snippets in the **scripts/** directory solve recurring sub-problems (e.g., data pre-processing, architecture modular implementation, fine-tuning) using the above-mentioned algorithms.  

### Projects 

The modules in the **projects/** directory contain core dynamical `classes` and `methods` design for the following problem types:
- patter recognition:
  - `classifier`: classification
  - `cluster`: clustering

### Notebooks (Jupyter)

The end-to-end analyses in the **notebooks/** directory provide methodogical documentation on solving a given tasks within certain machine learning paradigms:
- `01_classifier`: classification on small datasets by SVM and RF
- `02_classifier_nn`: classification using neural networks (ANN for tabular data, CNN for images)
- `03_dimensionality`: dimensionality reduction using PCA

## Installation

Clone the repository: ```$git clone https://github.com/samlipton/machine_learning.git```

Install dependencies: ```$pip install numpy scipy matplotlib tensorflow``` 

## Usage

Run the [Jupyter notebooks](https://github.com/samlipton/machine_learning/tree/master/notebooks) for exploratory analysis

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
