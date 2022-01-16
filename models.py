"""
This file defines the models used
"""

from sklearn import dummy, ensemble, linear_model, naive_bayes, tree
import xgboost
import lightgbm

CLASSIFICATION = [
    ("benchmark", dummy.DummyClassifier()),
    ("adaboost", ensemble.AdaBoostClassifier()),
    ("bagging", ensemble.BaggingClassifier()),
    ("extra-trees", ensemble.ExtraTreesClassifier()),
    ("gbc", ensemble.GradientBoostingClassifier()),
    ("random-forest", ensemble.RandomForestClassifier()),
    ("logreg", linear_model.LogisticRegression(
        penalty="none", solver="saga", random_state=0
    )),
    ("lasso", linear_model.LogisticRegression(
        penalty="l1", solver="saga", random_state=0
    )),
    ("ridge", linear_model.LogisticRegression(
        penalty="l2", solver="saga", random_state=0
    )),
    ("gnb", naive_bayes.GaussianNB()),
    ("bnb", naive_bayes.BernoulliNB()),
    ("dtc", tree.DecisionTreeClassifier()),
    ("xgboost", xgboost.XGBClassifier(
        eval_metric="logloss", use_label_encoder=False
    )),
    ("lgbm", lightgbm.LGBMClassifier())
]