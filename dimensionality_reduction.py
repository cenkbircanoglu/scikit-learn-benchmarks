# -*- coding: utf-8 -*-
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def reduce_dimension(name, x, n_components):
    algorithms = {
        'factor_analysis': FactorAnalysis(random_state=0,
                                          n_components=n_components),
        'fast_ica': FastICA(random_state=0, n_components=n_components),
        'nmf': Pipeline([('min_max', MinMaxScaler()), (
            'nmf', NMF(random_state=0, n_components=n_components))]),
        'pca': PCA(random_state=0, n_components=n_components),
        'sparse_pca': SparsePCA(random_state=0, n_components=n_components),
        'truncated_svd': TruncatedSVD(random_state=0,
                                      n_components=n_components)
    }
    return Pipeline([(name, algorithms.get(name)),
                     ('min_max', MinMaxScaler())]).fit_transform(x)
