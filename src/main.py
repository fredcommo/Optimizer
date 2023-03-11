# Inspired by:
# https://towardsdatascience.com/exploring-optuna-a-hyper-parameter-framework-using-logistic-regression-84bd622cd3a5

# Importing the Packages:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection

from sklearn import datasets
from sklearn import decomposition

import optuna

from optimizers import LogisticRegression_optimizer
from optimizers import RandomForestClassifier_optimizer
from optimizers import XGBClassifier_optimizer


############################################
# Other utils:
def clean_param_names(params):
    """
    Retrieve the original param names,
    so they can be passed to the best model
    """
    clean = lambda p: "_".join(p.split("_")[1:])
    return {clean(p): v for p, v in params.items()}


############################################
# Define an objective function to be maximized:
def objective(trial, X_train, y_train, model_list):

    # Setup values for the hyperparameters optimization:
    classifier_name = trial.suggest_categorical("classifier", model_list)
    classifier_optimizer = f"{classifier_name}_optimizer"
    classifier_obj = eval(classifier_optimizer)(trial)

    # Scoring method:
    score = model_selection.cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=5)

    # Return accuracy
    return score.mean()


def optimize(X_train, y_train, n_trials=5):
    """
    Run optimizer
    note: to pass args to the objective func, wrap it inside a lambda func + args
    and call the lambda func in study.optimize()
    """
    model_list = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]
    objective_func = lambda trial: objective(trial, X_train, y_train, model_list)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

    return study


def get_best_model(study):
    best_value = study.best_value
    best_params = study.best_params
    best_classifier = best_params.pop('classifier')
    best_params = clean_param_names(best_params)

    ############################################
    # Getting the best result
    print(f"\nBest accuracy: {best_value}")
    print(f"Best algorithm: {best_classifier}")
    print(f"Best parameters (ready to use): {best_params}\n")

    return best_classifier, best_params


def train_best_model(study, X_train, X_test, y_train, y_test):

    best_classifier, best_params = get_best_model(study)

    print(f"Running {best_classifier} as best classifier")
    print("Params:")
    print(best_params)

    classifier = eval(best_classifier)(**best_params)
    classifier.fit(X_train, y_train)

    # Print performances on test data
    print("Train accuracy: %.3f" % (np.mean(classifier.predict(X_train) == y_train)))
    print("Test accuracy: %.3f" % (np.mean(classifier.predict(X_test) == y_test)))

    return classifier


def show_cross_tab(observed, predictions):
    result = pd.crosstab(observed, predictions, rownames=["Observed"], colnames=["Predicted"])
    print("\nResults on test set:")
    print(result)


def plot(df):
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(df.x1, df.x2, c=df.y.astype('category').cat.codes, s=100, alpha=0.5)
    handles, labels = scatter.legend_elements()

    plt.xlabel("PC 1", fontsize=18)
    plt.ylabel("PC 2", fontsize=18)
    plt.legend(handles=handles, labels=labels, title="Status", fontsize=16)

    plt.show()


def main():
    ############################################
    # Grab a sklearn Classification dataset:
    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    ############################################
    # Run the optimizer:
    study = optimize(X_train, y_train)

    ############################################
    # Train the classifier and make predictions:
    classifier = train_best_model(study, X_train, X_test, y_train, y_test)
    predictions = classifier.predict(X_test)

    ############################################
    # Show results:
    show_cross_tab(y_test, predictions)

    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    new_x = pca.transform(X_test)
    df = pd.DataFrame({"x1": new_x[:,0], "x2": new_x[:,1], "y": predictions})
    plot(df)

if __name__ == "__main__":
    main()