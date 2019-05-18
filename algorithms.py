# -*- coding: utf-8 -*-
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree


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
        'gradient_boosting': ensemble.GradientBoostingClassifier()
    }
    clf = algorithms.get(name)
    clf.fit(x_tr, y_tr)
    tr_score = clf.score(x_tr, y_tr)
    score = clf.score(x_te, y_te)
    print(tr_score, score)
    return {name: {'test': score, "train": tr_score}}
