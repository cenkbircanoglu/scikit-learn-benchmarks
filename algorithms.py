# -*- coding: utf-8 -*-
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier


class CVParameters:
    ada_boost = {
        'algorithm': ['SAMME', 'SAMME.R'],
        'learning_rate': [i / 10. for i in range(1, 10, 1)],
        'n_estimators': list(range(10, 100, 10))
    }
    bagging = {
        'n_estimators': list(range(5, 50, 5)),
        'bootstrap_features': [0, 1]
    }

    extra_trees = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': list(range(5, 50, 5)),
        'warm_start': [1, 0]
    }

    random_forest = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': list(range(5, 50, 5)),
        'oob_score': [1, 0],
        'warm_start': [1, 0]
    }

    logistic_regression = {
        'tol': [1e-3 / i for i in range(10, 100, 10)],
        'C': [i / 10 for i in range(5, 15, 1)],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }

    passive_aggressive = {
        'tol': [1e-3 / i for i in range(10, 100, 10)],
        'early_stopping': [True, False],
        'loss': ['hinge', 'squared_hinge'],
        'warm_start': [1, 0]
    }

    ridge = {
        'alpha': [i / 10 for i in range(5, 15, 1)],
        'tol': [1e-3 / i for i in range(10, 100, 10)]
    }

    sgd = {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge',
                 'perceptron'],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'alpha': [i / 10000 for i in range(8, 12, 1)],
        'tol': [1e-3 / i for i in range(10, 100, 10)]
    }

    bernoulli = {
        'alpha': [i / 10 for i in range(1, 10, 1)],
    }

    gaussian = {
        'var_smoothing': [1e-9 / i for i in range(10, 100, 10)],
    }

    k_neighbors = {
        'n_neighbors': [i for i in range(3, 8, 1)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2, 3]
    }

    nearest_centroid = {
        'metric': ['euclidean', 'cosine', 'manhattan']
    }

    mlp = {
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'alpha': [0.0001 / i for i in range(10, 100, 10)],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'early_stopping': [True]
    }

    linear_svc = {
        'penalty': ['l2'],
        'multi_class': ['ovr', 'crammer_singer'],
        'tol': [1e-3 / i for i in range(10, 100, 10)],
        'C': [i / 10 for i in range(5, 15, 1)]
    }

    decision_tree = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }

    extra_tree = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }

    gradient_boosting = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': [i / 10. for i in range(1, 10, 1)],
        'criterion': ['friedman_mse'],
        'tol': [1e-4 / i for i in range(10, 100, 10)]
    }

    hist_gradient_boosting = {
        'l2_regularization': [0, 0.1],
        'tol': [1e-7 / i for i in range(10, 100, 10)]
    }


rstate = 0


def train_test(x_tr, y_tr, x_te, y_te, name):
    algorithms = {
        'ada_boost': ensemble.AdaBoostClassifier(),
        'bagging': ensemble.BaggingClassifier(),
        'extra_trees': ensemble.ExtraTreesClassifier(),
        'random_forest': ensemble.RandomForestClassifier(),
        'logistic_regression': linear_model.LogisticRegression(),
        'passive_aggressive': linear_model.PassiveAggressiveClassifier(),
        'ridge': linear_model.RidgeClassifier(),
        'sgd': linear_model.SGDClassifier(),
        'bernoulli': naive_bayes.BernoulliNB(),
        'gaussian': naive_bayes.GaussianNB(),
        'k_neighbors': neighbors.KNeighborsClassifier(),
        'nearest_centroid': neighbors.NearestCentroid(),
        'mlp': neural_network.MLPClassifier(),
        'linear_svc': svm.LinearSVC(),
        'decision_tree': tree.DecisionTreeClassifier(),
        'extra_tree': tree.ExtraTreeClassifier(),
        'gradient_boosting': ensemble.GradientBoostingClassifier(),
        'hist_gradient_boosting': HistGradientBoostingClassifier()
    }
    clf = GridSearchCV(algorithms.get(name), getattr(CVParameters, name),
                       cv=5, n_jobs=-1)
    clf.fit(x_tr, y_tr)
    print(clf.best_params_)
    print(clf.best_score_)
    tr_score = clf.score(x_tr, y_tr)
    score = clf.score(x_te, y_te)
    tr_fscore = f1_score(y_tr, clf.predict(x_tr))
    fscore = f1_score(y_te, clf.predict(x_te))
    print(tr_score, score, tr_fscore, fscore)
    return {name: {'test': score, "train": tr_score, 'f1_test': fscore,
                   'f1_train': tr_fscore}}
