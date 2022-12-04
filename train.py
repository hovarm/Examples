import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, auc, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import pickle as pkl

models = [LogisticRegression(C=1.0, class_weight=None, dual=False,
                             fit_intercept=True,
                             intercept_scaling=1, l1_ratio=None,
                             max_iter=10000,
                             multi_class='auto', n_jobs=None, penalty='l2',
                             random_state=None, solver='lbfgs', tol=0.0001,
                             verbose=0,
                             warm_start=False),
          LinearDiscriminantAnalysis(n_components=None, priors=None,
                                     shrinkage='auto',
                                     solver='lsqr', store_covariance=False,
                                     tol=0.0001),
          QuadraticDiscriminantAnalysis(),
          DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None,
                                 criterion='gini',
                                 max_depth=6, max_features=None,
                                 max_leaf_nodes=11,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None,
                                 min_samples_leaf=1, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0,
                                 presort='deprecated',
                                 random_state=None, splitter='best'),
          GaussianNB(priors=None, var_smoothing=0.3511191734215131),
          SVC(C=100, break_ties=False, cache_size=200, class_weight=None,
              coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma=0.01,
              kernel='rbf',
              max_iter=-1, probability=True, random_state=None, shrinking=True,
              tol=0.001, verbose=False)]

clf_list = [(i.__class__.__name__, i) for i in models]

ensemble_models = [
    VotingClassifier(estimators=clf_list, voting='soft'),
    StackingClassifier(estimators=clf_list,
                       final_estimator=LogisticRegression()),
    BaggingClassifier(base_estimator=models[0],
                      n_estimators=100, n_jobs=-1),
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                               max_depth=10),
    AdaBoostClassifier(n_estimators=100)
]

thresholds = [0.5, 0.5, 0.65, 0.5, 0.45, 0.5]


# fill gender missing values by DT classifier
def fill_missing_class(col_name, df):
    X = df[df.columns.difference(['In-hospital_death'])]
    cols = []
    for col in X.columns:
        if df[col].isnull().sum() == 0:
            cols.append(col)
    idx = X[X[col_name].isna() == True].index
    X_test = X.loc[idx, cols]
    X_train = X[cols].drop(index=idx)
    y_train = X[col_name].drop(index=idx)
    pred = DecisionTreeClassifier().fit(X_train, y_train).predict(X_test)
    df[col_name] = df[col_name].fillna(pd.Series(pred, index=idx))
    return df


# fill weight missing values by Linear Regression
def fill_missing_value(corr_features, col_name, df):
    X = df[df.columns.difference(['In-hospital_death'])]
    idx = df[df[col_name].isna()].index
    X_test = X.loc[idx, corr_features]
    X_train = X[corr_features].drop(index=idx)
    y_train = df[col_name].drop(index=idx)
    pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    df[col_name] = df[col_name].fillna(pd.Series(pred, index=idx))
    return df


def clean_data(data, mod='train'):
    if mod == "train":
        df = data.drop(['Length_of_stay', 'Survival', 'recordid'], axis=1)
        y = df['In-hospital_death']
        X = df[df.columns.difference(['In-hospital_death'])]
    else:
        try:
            X = data[data.columns.difference(['In-hospital_death',
                                              'Length_of_stay',
                                              'Survival',
                                              'recordid'])]
        except Exception as ex:
            print(ex)


    df = fill_missing_class("Gender", X)
    corr_features = ['Age', 'Gender']
    df = fill_missing_value(corr_features, 'Weight', df)
    corr_features = ['Age', 'Gender', 'Weight']
    df = fill_missing_value(corr_features, 'Height', df)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            idx = X[X[col].isna() == False].index
            x = df[col][idx]
            df[col].fillna(np.median(x), inplace=True)

    dummies = ['Gender', 'CCU', 'CSRU', 'SICU', 'MechVentLast8Hour']
    norm = MinMaxScaler().fit(X[X.columns.difference(dummies)])
    X_normalised = norm.transform(X[X.columns.difference(dummies)])
    x_norm = pd.DataFrame(X_normalised,
                          columns=X.columns.difference(dummies),
                          index=X.index)
    X = pd.concat([x_norm, X[dummies]], axis=1)

    if mod == 'train':
        return X, y
    else:
        return X, None


def statistics(df):
    X, y = clean_data(df)
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)

    Accuracy, Precision, Recall, F1_score, AUC = list(), list(), list(), list(), list()

    for i in models:
        i.fit(xTrain, yTrain)

    for i in ensemble_models:
        i.fit(xTrain, yTrain)

    for i, m in enumerate(models):
        predictions = np.where(
            m.predict_proba(xTest)[:, 1] > thresholds[i], 1, 0)
        fpr, tpr, _ = roc_curve(yTest, predictions)
        F1_score.append(f1_score(yTest, predictions))
        AUC.append(roc_auc_score(yTest, m.predict_proba(xTest)[:, 1]))

    for i, m in enumerate(ensemble_models):
        predictions = m.predict(xTest)
        F1_score.append(f1_score(yTest, predictions))
        AUC.append(roc_auc_score(yTest, m.predict_proba(xTest)[:, 1]))

    final_models = models + ensemble_models
    for i, m in enumerate(final_models):
        print("\n", m.__class__.__name__)
        print("f1_score: ", F1_score[i])
        print("AUC: ", AUC[i])

    f1_model = final_models[np.argmax(F1_score)]
    auc_model = final_models[np.argmax(AUC)]
    X, y = clean_data(df)

    f1_model.fit(X, y)
    auc_model.fit(X, y)

    pkl.dump(f1_model,
             open(f1_model.__class__.__name__,
                  "wb"))
    pkl.dump(auc_model,
             open(auc_model.__class__.__name__,
                  "wb"))


def main():
    df = pd.read_csv("Survival_dataset.csv")
    statistics(df)

if __name__ == "__main__":
    main()
