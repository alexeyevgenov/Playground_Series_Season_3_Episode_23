from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import optuna
from first_touch import cross_val
import pickle


train_data = pd.read_csv("data/train.csv", index_col="id")
test_data = pd.read_csv("data/test.csv", index_col="id")


def objective(trial):
    params = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
              # 'max_depth': trial.suggest_int('max_depth', 3, 8),
              # 'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
              # 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
              'l2_regularization': trial.suggest_float('l2_regularization', 0, 10),
              # 'max_bins': trial.suggest_int('max_bins', 50, 255),
              # 'early_stopping': trial.suggest_categorical('early_stopping', [True, False])
              }

    return cross_val(HistGradientBoostingClassifier(**params, random_state=1), "")


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # print('Best parameters:')
    # print(study.best_trial.params)

    model = HistGradientBoostingClassifier(**study.best_trial.params, random_state=1)
    # X_train = train_data.copy()
    # y_train = X_train.pop('defects')
    # model.fit(X_train, y_train)
    pickle.dump(study.best_trial.params, open("Hist_Grad_Boost_Classifier_Model.pkl", "wb"))
