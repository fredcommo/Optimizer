# Original code at:
# https://towardsdatascience.com/exploring-optuna-a-hyper-parameter-framework-using-logistic-regression-84bd622cd3a5

# Importing the Packages:
import functools

import optuna
import numpy as np

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection

from sklearn import datasets

from optimizers import LogisticRegression_optimizer
from optimizers import RandomForestClassifier_optimizer
from optimizers import XGBClassifier_optimizer


############################################
# Other utils:
def clean_param_names(params):
    clean = lambda p: "_".join(p.split("_")[1:])
    return {clean(p): v for p, v in params.items()}


############################################
# Define an objective function to be maximized:
def objective(trial, X_train, y_train):

    model_list = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]

    # Setup values for the hyperparameters optimization:
    classifier_name = trial.suggest_categorical("classifier", model_list)
    classifier_optimizer = f"{classifier_name}_optimizer"
    classifier_obj = eval(classifier_optimizer)(trial)

    # Scoring method:
    # X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=123)
    score = model_selection.cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=5)

    # Return accuracy
    return score.mean()


def run_best_model(X_train, X_test, y_train, y_test, best_classifier, best_params):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    print(f"Running {best_classifier} as best classifier")
    print("Params:")
    print(best_params)

    classifier = eval(best_classifier)(**best_params)
    classifier.fit(X_train, y_train)

    # Print performances on test data
    print("Train accuracy: %.3f" % (np.mean(classifier.predict(X_train) == y_train)))
    print("Test accuracy: %.3f" % (np.mean(classifier.predict(X_test) == y_test)))

    return classifier


def main(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    ############################################
    # Running optimizer
    # to add args to the objective: Wrap the objective inside a lambda and call objective inside it
    objective_func = lambda trial: objective(trial, X_train, y_train)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=100, show_progress_bar=True)

    best_value = study.best_value
    best_params = study.best_params
    best_classifier = best_params.pop('classifier')

    # Retrieve the original param names,
    # so they can be directly passed to the best algo
    best_params = clean_param_names(best_params)

    ############################################
    # Getting the best result
    print()
    print(f"Best accuracy: {best_value}")
    print(f"Best algorithm: {best_classifier}")
    print(f"Best parameters (ready to use): {best_params}")
    print()

    ############################################
    # Running the best model
    classifier = run_best_model(X_train, X_test, y_train, y_test, best_classifier, best_params)

    return classifier


if __name__ == "__main__":

    ############################################
    # Grabbing a sklearn Classification dataset:
    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    classifier = main(X, y)
