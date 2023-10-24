from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
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
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 90, 120),
        # 'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        # 'max_depth': trial.suggest_int('max_depth', 10, 100),
        # 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 90, 120),
        # 'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        # 'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    return cross_val(make_pipeline(ColumnTransformer([('drop', 'drop',
                                                       ['iv(g)', 't', 'b', 'n', 'lOCode', 'v',
                                                        'branchCount', 'e', 'i', 'lOComment'])],
                                                     remainder='passthrough'),
                                   FunctionTransformer(np.log1p),
                                   ExtraTreesClassifier(n_estimators=100,
                                                        min_samples_leaf=100,
                                                        max_features=1.0,
                                                        random_state=1)))


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)

    # print('Best parameters:')
    # print(study.best_trial.params)

    model = make_pipeline(ColumnTransformer([('drop', 'drop',
                                              ['iv(g)', 't', 'b', 'n', 'lOCode', 'v',
                                               'branchCount', 'e', 'i', 'lOComment'])],
                                            remainder='passthrough'),
                          FunctionTransformer(np.log1p),
                          ExtraTreesClassifier(**study.best_trial.params, random_state=1))
    # X_train = train_data.copy()
    # y_train = X_train.pop('defects')
    # model.fit(X_train, y_train)
    pickle.dump(model, open("Extra_Trees_Classifier_Pipeline.pkl", "wb"))
