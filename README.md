# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about Bank Marketing campaigns based on phone calls. We seek to predict if the contacted client has subscribed to a term deposit (yes/no). 

The best performing model waas
**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline

First a training script is prepared that trains a model for a single run on the data, using the following steps:
- Load data from URL 
- Clean and prepare data
- Split into training and test datasets
- Fit a Logistic Regression with given parameters and save/log results

Next the model in train.py is used by HyperDrive to perform hyperparameter tuning:
- Randomly sample regression strength and max iterations to find best hyperparameters in acceptable time (grid search is exhaustive but time/resource consuming)
- Choose an early stopping policy to not waste time and compute on bad runs
- Run train.py for #n number of runs to get best logistic regression model

I ran 50 runs and the best Logistic regression had an accuracy of 0.911

## AutoML
The automated ML resulted in the best model ...
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
In comparison..the automl achieved a better accuracy than the tuned logistic regression. This can be due to better (automated) feature engineering, data balancing, and algorithm comparison/selection.

## Future work
Proper data inspection and preparation. For example balancing classes, removing highly correlated features, applying data transformations.
