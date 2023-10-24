import optuna
from sklearn.ensemble import VotingClassifier, HistGradientBoostingClassifier
import pickle
from extra_trees import cross_val
import pandas as pd

train_data = pd.read_csv("data/train.csv", index_col="id")
test_data = pd.read_csv("data/test.csv", index_col="id")
et_pipeline = pickle.load(open("Extra_Trees_Classifier_Pipeline.pkl", "rb"))
gb_params = pickle.load(open("Hist_Grad_Boost_Classifier_Model.pkl", "rb"))
lr_pipeline = pickle.load(open("Logistic_Regression_Pipeline.pkl", "rb"))


def objective(trial):
    et_weight = trial.suggest_float("et_weight", 0, 2)
    gb_weight = trial.suggest_float("gb_weight", 0, 2)
    lr_weight = trial.suggest_float("lr_weight", 0, 2)

    model = ensemble([et_weight, gb_weight, lr_weight])

    return cross_val(model)


def ensemble(weights):
    ens = VotingClassifier(
        [('gb', HistGradientBoostingClassifier(**gb_params)),
         ('et', et_pipeline),
         ('lr', lr_pipeline)],
        voting='soft',
        weights=weights)
    return ens


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(study.best_trial.params)

ens = ensemble(list(study.best_trial.params.values()))
ens.fit(train_data.iloc[:, :-1], train_data.defects)
y_pred = ens.predict_proba(test_data)[:, 1]
submission = pd.Series(y_pred, index=test_data.index, name='defects')
submission.to_csv('submission.csv')
pickle.dump(ens, open("Voting_Classifier_Ensemble.pkl", "wb"))
