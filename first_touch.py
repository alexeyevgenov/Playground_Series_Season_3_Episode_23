import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, \
    VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from datetime import datetime
from functools import partial

from sklearn.svm import LinearSVC

np.random.RandomState(41)

train_data = pd.read_csv("data/train.csv", index_col="id")
test_data = pd.read_csv("data/test.csv", index_col="id")
collinearity_threshold = 0.85


def cross_val(model, label):
    """
    Cross-validate the model with a StratifiedKFold
    """
    start_time = datetime.now()
    skf = StratifiedKFold(shuffle=True, random_state=1)
    auc_list = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(train_data, train_data.defects)):
        X_train = train_data.iloc[idx_tr]
        X_valid = train_data.iloc[idx_va]
        y_train = X_train.pop('defects')
        y_valid = X_valid.pop('defects')
        model.fit(X_train, y_train)
        try:
            y_valid_pred = model.predict_proba(X_valid)[:, 1]
        except AttributeError:  # 'LinearSVC' object has no attribute 'predict_proba'
            y_valid_pred = model.decision_function(X_valid)
        auc = roc_auc_score(y_valid, y_valid_pred)
        auc_list.append(auc)
    auc = np.array(auc_list).mean()
    execution_time = datetime.now() - start_time
    if len(label) != 0:
        print(f"# AUC {auc:.5f}   time={str(execution_time)[-15:-7]}   {label}")
    return auc


def remove_highly_collinear_variables(df: pd.DataFrame, collinearity_threshold: float) -> pd.DataFrame:
    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > collinearity_threshold)]
    print(f"There are defined {len(to_drop)} features with correlation greater than {collinearity_threshold}: {to_drop}"
          )
    df.drop(to_drop, axis=1, inplace=True)
    return df


def remove_highly_collinear_features_from_array(X, threshold):

    corr_matrix = np.corrcoef(X, rowvar=False)

    upper = np.triu(corr_matrix, k=1)

    to_drop = []
    for i in range(upper.shape[1]):
        if np.any(upper[:, i] > threshold):
            to_drop.append(i)

    print(f"Found {len(to_drop)} from {X.shape[1]} highly collinear features to remove: {to_drop}")

    return np.delete(X, to_drop, axis=1)


def logistic_regression():
    for C in np.logspace(-2, 1, 9):
        auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                      PolynomialFeatures(2, include_bias=False),
                                      StandardScaler(),
                                      LogisticRegression(dual=False, C=C,
                                                         class_weight='balanced',
                                                         max_iter=1500,
                                                         random_state=1,
                                                         solver='newton-cholesky')),
                        f'Poly-LogisticRegression {C=:.2g}')


def nude_logistic_regression():
    for C in np.logspace(-2, 1, 9):
        auc = cross_val(LogisticRegression(dual=False, C=C,
                                           class_weight='balanced',
                                           max_iter=1500,
                                           random_state=1,
                                           # solver='newton-cholesky'
                                           ),
                        f'Poly-LogisticRegression {C=:.2g}')


def logistic_regression_kernel_approximation():
    n_components = 400
    for C in np.logspace(-3, -2, 9):  # C=0.0024
        auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                      Nystroem(n_components=n_components, random_state=10),  # kernel approximation
                                      StandardScaler(),
                                      LogisticRegression(dual=False, C=C,
                                                         class_weight='balanced',
                                                         max_iter=1500,
                                                         random_state=1,
                                                         solver='newton-cholesky')),
                        f'Nyström-LogisticRegression {n_components=} {C=:.2g}')


def extra_trees_classifier():
    auc = cross_val(make_pipeline(ColumnTransformer(
        [('drop', 'drop', ['iv(g)', 't', 'b', 'n', 'lOCode', 'v', 'branchCount', 'e', 'i', 'lOComment'])],
        remainder='passthrough'),
        FunctionTransformer(np.log1p),
        ExtraTreesClassifier(n_estimators=100,
                             min_samples_leaf=100,
                             max_features=1.0,
                             random_state=1)),
        f"Feature-selection-ET")


def random_forest_classifier():
    for min_samples_leaf in [100, 150, 200, 250, 300]:  # min_samples_leaf = 150
        auc = cross_val(RandomForestClassifier(n_estimators=100,
                                               min_samples_leaf=min_samples_leaf,
                                               max_features=1.0,
                                               random_state=1
                                               ),
                        f"RF {min_samples_leaf=}"
                        )


def k_neighbors_classifier():
    for n_neighbors in range(200, 800, 100):
        auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                      StandardScaler(),
                                      KNeighborsClassifier(n_neighbors=n_neighbors,
                                                           weights='distance')),
                        f"KNN {n_neighbors=}")


def linear_svc():
    for C in np.logspace(-4, -1, 4):
        auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                      PolynomialFeatures(2, include_bias=False),
                                      StandardScaler(),
                                      LinearSVC(dual=False, C=C)),
                        f'Poly-LinearSVC {C=:.2g}')


def hist_gradient_boosting_classifier():
    auc = cross_val(HistGradientBoostingClassifier(random_state=1),
                    f"HistGradientBoostingClassifier")


def lightgbm():
    for num_leaves in [20, 50, 100, 150, 200]:
        auc = cross_val(LGBMClassifier(num_leaves=num_leaves,
                                       n_estimators=100,
                                       subsample=0.8,
                                       colsample_bytree=0.8,
                                       reg_alpha=0.1,
                                       reg_lambda=0.1,
                                       max_depth=-1,
                                       objective='binary',
                                       metric='auc',
                                       random_state=1),
                        f'Poly-LinearSVC {num_leaves=:.2g}')


def multi_layer_perceptron():
    correlated_data_remover = partial(remove_highly_collinear_features_from_array,
                                      threshold=collinearity_threshold)
    for hidden_layer_sizes in [(100,), (200, 100), (300, 100, 10)]:
        auc = cross_val(make_pipeline(FunctionTransformer(np.log1p),
                                      PolynomialFeatures(2, include_bias=False),
                                      # FunctionTransformer(correlated_data_remover),
                                      # RFE(LinearRegression(), n_features_to_select=30),
                                      PCA(n_components=20),
                                      StandardScaler(),
                                      MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                    max_iter=5000,
                                                    # early_stopping=True
                                                    verbose=True)
                                      ),
                        f'Multi-Layer-Perceptron Classifier {hidden_layer_sizes}')


def ensemble():
    ens = VotingClassifier(
        [('hgb', HistGradientBoostingClassifier(random_state=1)),
         ('et', make_pipeline(ColumnTransformer([('drop', 'drop',
                                                  ['iv(g)', 't', 'b', 'n', 'lOCode', 'v',
                                                   'branchCount', 'e', 'i', 'lOComment'])],
                                                remainder='passthrough'),
                              FunctionTransformer(np.log1p),
                              ExtraTreesClassifier(n_estimators=100,
                                                   min_samples_leaf=100,
                                                   max_features=1.0,
                                                   random_state=1))),
         ('ny', make_pipeline(FunctionTransformer(np.log1p),
                              Nystroem(n_components=400, random_state=1),
                              StandardScaler(),
                              LogisticRegression(dual=False, C=0.0032,
                                                 max_iter=1500,
                                                 random_state=1)))],
        voting='soft',
        weights=[0.4, 0.4, 0.2])
    auc = cross_val(ens, 'Ensemble(HGB+ET+NY)')
    return ens


if __name__ == "__main__":
    # logistic_regression()
    # extra_trees_classifier()
    multi_layer_perceptron()
    # random_forest_classifier()

    # nude_logistic_regression()

    # ens = ensemble()
    # ens.fit(train_data.iloc[:, :-1], train_data.defects)
    # y_pred = ens.predict_proba(test_data)[:, 1]
    # submission = pd.Series(y_pred, index=test_data.index, name='defects')
    # submission.to_csv('submission.csv')
