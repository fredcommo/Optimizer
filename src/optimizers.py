# Original code at:
# https://towardsdatascience.com/exploring-optuna-a-hyper-parameter-framework-using-logistic-regression-84bd622cd3a5

# Importing the Packages:
import functools

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


############################################
# Decorator:
def optimizer(model):
    def deco_optim(parameterizer):
        @functools.wraps(parameterizer)
        def wrapper_optim(*args):
            params = parameterizer(*args)
            return model(**params)
        return wrapper_optim
    return deco_optim


############################################
# Set trial parameters for each model:
@optimizer(LogisticRegression)
def LogisticRegression_optimizer(trial):
    params = {
        "C": trial.suggest_float("logReg_C", 1e-3, 1e3, log=True),
        "max_iter": 20000
        }
    return params

@optimizer(RandomForestClassifier)
def RandomForestClassifier_optimizer(trial):
    params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 2, 32, log=True),
        "max_depth": trial.suggest_int("rf_max_depth", 10, 1000)
        }
    return params

@optimizer(XGBClassifier)
def XGBClassifier_optimizer(trial):
    params = {
        "n_estimators": trial.suggest_int('xgb_n_estimators', 10, 2000),
        "max_depth": trial.suggest_int('xgb_max_depth', 2, 8),
        "colsample_bylevel": trial.suggest_float("xgb_colsample_bylevel", 0.1, 1),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.1),
        "alpha": trial.suggest_float("xgb_alpha", 0.1, 1),
        "lambda": trial.suggest_float("xgb_lambda", 0.1, 1)
        }
    return params
