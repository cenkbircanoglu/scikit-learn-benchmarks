# -*- coding: utf-8 -*-
from sklearn.decomposition import FactorAnalysis, FastICA, NMF, PCA, \
    TruncatedSVD, SparsePCA


def reduce_dimension(name, x, n_components):
    algorithms = {
        'factor_analysis': FactorAnalysis(random_state=0,
                                          n_components=n_components),
        'fact_ica': FastICA(random_state=0, n_components=n_components),
        'nmf': NMF(random_state=0, n_components=n_components),
        'pca': PCA(random_state=0, n_components=n_components),
        'sparse_pca': SparsePCA(random_state=0, n_components=n_components),
        'truncated_svd': TruncatedSVD(random_state=0,
                                      n_components=n_components)
    }
    return algorithms.get(name).fit_transform(x)
