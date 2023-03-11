# Original code at:
# https://towardsdatascience.com/exploring-optuna-a-hyper-parameter-framework-using-logistic-regression-84bd622cd3a5

# Importing the Packages:
import functools

import optuna
# import pandas as pd
import numpy as np
# import re

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from optimizers import logReg_optimizer
from optimizers import RF_optimizer
from optimizers import xgb_optimizer


############################################
# Other utils:
def clean_param_names(params):
    clean = lambda p: "_".join(p.split("_")[1:])
    return {clean(k): v for k, v in params.items()}


############################################
# Define an objective function to be maximized:
def objective(trial):

    classifier_name = trial.suggest_categorical("classifier", ["LogReg", "RandomForest", "XGBClassifier"])
    
    # Setup values for the hyperparameters:
    if classifier_name == 'LogReg':
        classifier_obj = logReg_optimizer(trial)

    elif classifier_name == "RandomForest":
        classifier_obj = RF_optimizer(trial)

    else:
        classifier_obj = xgb_optimizer(trial)

    # Scoring method:
    score = model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=5)

    # Return accuracy
    return score.mean()


def run_best_model(X, y, classifier_name, best_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    # Initialize classifier
    if classifier_name == 'LogReg':
        classifier = LogisticRegression(**best_params)

    elif classifier_name == "RandomForest":
        classifier = RandomForestClassifier(**best_params)

    else:
        classifier = XGBClassifier(**best_params)

    # Train
    classifier.fit(X_train, y_train)

    # Print performances on test data
    print("Train accuracy: %.3f" % (np.mean(classifier.predict(X_train) == y_train)))
    print("Test accuracy: %.3f" % (np.mean(classifier.predict(X_test) == y_test)))

    return classifier


def main(X, y):
    ############################################
    # Running optimizer
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    best_value = study.best_value
    best_params = study.best_params
    best_classifier_name = best_params.pop('classifier')

    # Retrieve the original param names,
    # so they can be directly passed to the best algo
    best_params = clean_param_names(best_params)

    ############################################
    # Getting the best result
    print()
    print(f"Best accuracy: {best_value}")
    print(f"Best algorithm: {best_classifier_name}")
    print(f"Best parameters (ready to use): {best_params}")
    print()

    ############################################
    # Running the best model
    classifier = run_best_model(X, y, best_classifier_name, best_params)

    return classifier


if __name__ == "__main__":

    ############################################
    # Grabbing a sklearn Classification dataset:
    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    classifier = main(X, y)
