# -*- coding: utf-8 -*-
from sklearn.datasets import load_breast_cancer, fetch_covtype, load_digits, \
    load_iris, fetch_kddcup99, fetch_lfw_people, openml, \
    fetch_20newsgroups_vectorized, fetch_olivetti_faces, fetch_rcv1, load_wine


def load(name):
    cov_type = fetch_covtype()
    news_groups = fetch_20newsgroups_vectorized(subset='all')
    olivetti_faces = fetch_olivetti_faces()
    databases = {
        'breast_cancer': load_breast_cancer(return_X_y=True),
        'cov_type': (cov_type.data, cov_type.target),
        'digits': load_digits(return_X_y=True),
        'iris': load_iris(return_X_y=True),
        'lfw': fetch_lfw_people(return_X_y=True),
        'mnist': openml.fetch_openml('mnist_784', version=1, return_X_y=True),
        'news_groups': (news_groups.data, news_groups.target),
        'olivetti_faces': (olivetti_faces.data, olivetti_faces.target),
        'rcv1': fetch_rcv1(random_state=0, return_X_y=True),
        'wine': load_wine(return_X_y=True)
    }
    return databases.get(name)
