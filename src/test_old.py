# Original code at:
# https://towardsdatascience.com/exploring-optuna-a-hyper-parameter-framework-using-logistic-regression-84bd622cd3a5

# Importing the Packages:
import optuna
import pandas as pd
import numpy as np
import re

from xgboost import XGBClassifier

from sklearn import linear_model
from sklearn import ensemble
from sklearn import datasets
from sklearn import model_selection

from sklearn.model_selection import train_test_split


############################################
# Grabbing a sklearn Classification dataset:
X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)


############################################
# Set trial parameters for each model:
def _logReg_params(trial):
    params = {
        "C": trial.suggest_float("logReg_C", 1e-10, 1e10, log=True),
        "max_iter": 1000
        }
    return params

def _RF_params(trial):
    params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 2, 32, log=True),
        "max_depth": trial.suggest_int("rf_max_depth", 10, 1000)
        }
    return params

def _xgb_params(trial):
    params = {
        "n_estimators": trial.suggest_int('xgb_n_estimators', 20, 1000),
        "max_depth": trial.suggest_int('xgb_max_depth', 2, 5),
        "colsample_bylevel": trial.suggest_float("xgb_colsample_bylevel", 0.1, 1),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
        "alpha": trial.suggest_float("xgb_alpha", 0.001, 10),
        "lambda": trial.suggest_float("xgb_lambda", 0.001, 10)
        }
    return params

def clean_param_names(params):
    clean = lambda p: "_".join(p.split("_")[1:])
    return {clean(k): v for k, v in best_params.items()}


############################################
# Define an objective function to be maximized:
def objective(trial):

    classifier_name = trial.suggest_categorical("classifier", ["LogReg", "RandomForest", "XGBClassifier"])
    
    # Setup values for the hyperparameters:
    if classifier_name == 'LogReg':
        params = _logReg_params(trial)
        classifier_obj = linear_model.LogisticRegression(**params)

    elif classifier_name == "RandomForest":
        params = _RF_params(trial)
        classifier_obj = ensemble.RandomForestClassifier(**params)

    else:
        params = _xgb_params(trial)
        classifier_obj = XGBClassifier(**params)

    # Scoring method:
    score = model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=5)

    # Return accuracy
    return score.mean()

def run_best_model(X, y, classifier_name, best_params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    if classifier_name == 'LogReg':
        classifier = linear_model.LogisticRegression(**best_params)

    elif classifier_name == "RandomForest":
        classifier = ensemble.RandomForestClassifier(**best_params)

    else:
        classifier = XGBClassifier(**best_params)

    classifier.fit(X_train, y_train)

    print("Train accuracy: %.3f" % (np.mean(classifier.predict(X_train) == y_train)))
    print("Test accuracy: %.3f" % (np.mean(classifier.predict(X_test) == y_test)))

    return classifier

############################################
# Running optimizer
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)


############################################
# Getting the best result
print()

best_params = study.best_params
best_classifier_name = best_params.pop('classifier')

# Retrieve the original param names,
# so they can be directly passed to the best algo
best_params = clean_param_names(best_params)

print(f"Best accuracy: {study.best_value}")
print(f"Best algorithm: {best_classifier_name}")
print(f"Best parameters (ready to use): {best_params}")


############################################
# Running the best model
classifier = run_best_model(X, y, best_classifier_name, best_params)