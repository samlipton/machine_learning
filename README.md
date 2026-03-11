# Machine Learning Projects 

This repository contains a collection of machine learning experiments organized around a simple and reproducible project structure. 
The goal is to develop machine learning workflows that remain readable, modular, and easy to extend across multiple datasets and tasks. 
The repository follows a lightweight structure inspired by the project organization proposed by Eric Ma (see the reference structure here: https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510). 
Such structured layouts are commonly used to improve maintainability and reproducibility in machine learning projects by clearly separating data, exploratory work, and reusable code. :contentReference[oaicite:0]{index=0} 

## Philosophy 

Machine learning work often starts as exploratory experimentation and gradually evolves into structured and reusable code. 
This repository follows a workflow designed to reflect that process: 
1. **Scripts first** Core functionality is implemented as reusable Python scripts.
2. **Notebooks second** These scripts are then used inside notebooks to document experiments, results, and insights.
3. **Projects as thematic units** Each project explores a specific machine learning theme across one or several datasets.
4. **Datasets as reusable resources** Multiple datasets can be reused across different projects.

## Repository Structure 

The repository is organized around four main directories: 
``` machine_learning/ │ ├── data/ │ Storage for datasets used across projects. │ ├── scripts/ │ Reusable Python scripts implementing the core functionality │ (data processing, model training, evaluation, etc.). │ ├── notebooks/ │ Jupyter notebooks that demonstrate experiments and results │ using the scripts developed in the repository. │ └── projects/ Project-specific material organized by theme. ``` 

### Directory roles 
**data/** Contains the datasets used throughout the repository. Datasets may originate from external sources or from libraries such as `scikit-learn`. 
**scripts/** Contains reusable building blocks of the machine learning workflows. Typical content may include: - data loading utilities - preprocessing pipelines - model training functions - evaluation tools These scripts represent the stable and reusable part of the codebase. 
**notebooks/** Jupyter notebooks serve two main purposes: - exploratory analysis - documentation of experiments and results They import functionality from the `scripts` directory rather than reimplementing it. 
**projects/** Projects group together experiments related to a specific topic or research question. 

## Current Projects 

### Classification on Scikit-Learn Datasets 

The first project focuses on **classification tasks using datasets available in `scikit-learn`**. 

Objectives include: 
- applying multiple machine learning models
- comparing model performance across datasets
- exploring preprocessing and hyperparameter choices
- building reproducible experimental pipelines

The goal is not only to evaluate models but also to develop reusable workflows that can later be applied to other datasets and problems. 

## Typical Workflow 

A typical workflow in this repository follows these steps: 
1. Select a dataset from `data/` or `scikit-learn`.
2. Implement reusable functionality in `scripts/`.
3. Use notebooks to explore models and document results.
4. Organize the work within a project in `projects/`.

This workflow encourages separation between: - **reusable code** (scripts) - **experimentation and analysis** (notebooks) - **project-level organization** (projects) 
