from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import optuna
import pickle
from joblib import Parallel, delayed

train_data = pd.read_csv("data/train.csv", index_col="id")
test_data = pd.read_csv("data/test.csv", index_col="id")


def cross_val(model):
    """
    Cross-validate the model with a StratifiedKFold
    """

    skf = StratifiedKFold(shuffle=True, random_state=1)
    auc_list = Parallel(n_jobs=-2)(delayed(cross_val_fold)(model, train_data, idx_tr, idx_va)
                                   for idx_tr, idx_va in skf.split(train_data, train_data.defects))
    auc = np.array(auc_list).mean()
    return auc


def cross_val_fold(model, X, idx_train, idx_val):
    X_train = X.iloc[idx_train]
    X_valid = X.iloc[idx_val]
    y_train = X_train.pop('defects')
    y_valid = X_valid.pop('defects')

    model.fit(X_train, y_train)
    y_valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_valid_pred)
    return auc


def objective(trial):
    params = {'C': trial.suggest_float('C', 0.001, 0.005),
              # 'penalty': ['l1', 'l2', 'elasticnet'],
              # 'l1_ratio': optuna.uniform(0, 1),
              'max_iter': trial.suggest_int('max_iter', 1000, 2000),
              # 'tol': optuna.loguniform(1e-5, 1e-2),
              # 'class_weight': [None, 'balanced']

              }

    return cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                   Nystroem(n_components=400, random_state=1),
                                   StandardScaler(),
                                   LogisticRegression(**params, random_state=1)))


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # print('Best parameters:')
    # print(study.best_trial.params)

    model = make_pipeline(FunctionTransformer(np.log1p),
                          Nystroem(n_components=400, random_state=1),
                          StandardScaler(),
                          LogisticRegression(**study.best_trial.params, random_state=1))
    X_train = train_data.copy()
    y_train = X_train.pop('defects')
    model.fit(X_train, y_train)
    pickle.dump(study.best_trial.params, open("Logistic_Regression_Pipeline.pkl", "wb"))
