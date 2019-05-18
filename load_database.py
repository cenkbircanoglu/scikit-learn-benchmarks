# -*- coding: utf-8 -*-
from operator import itemgetter

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_kddcup99
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_rcv1
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import openml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils import LazyDict


def load_kddcup99():
    X, y = fetch_kddcup99(shuffle=1, return_X_y=True,
                          percent10=False)
    categorical_features = [1, 2, 3]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)])
    return preprocessor.fit_transform(X), LabelEncoder().fit_transform(y)


def load(name):
    """
    Load the database from Lazy Initialized Dictionary with its known name.
    :param name: Name of database
    :return: tuple(X, y)
    """
    databases = LazyDict({
        'breast_cancer': lambda: load_breast_cancer(return_X_y=True),
        'cov_type': lambda: itemgetter('data', 'target')(fetch_covtype()),
        'digits': lambda: load_digits(return_X_y=True),
        'iris': lambda: load_iris(return_X_y=True),
        'kddcup99': lambda: load_kddcup99(),
        'lfw': lambda: fetch_lfw_people(return_X_y=True),
        'mnist': lambda: openml.fetch_openml('mnist_784', version=1,
                                             return_X_y=True),
        'news_groups': lambda: itemgetter('data', 'target')(
            fetch_20newsgroups_vectorized(subset='all')),
        'olivetti_faces': lambda: itemgetter('data', 'target')(
            fetch_olivetti_faces()),
        'rcv1': lambda: fetch_rcv1(random_state=0, return_X_y=True),
        'wine': lambda: load_wine(return_X_y=True)
    })
    return databases.get(name)
