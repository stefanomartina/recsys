""" @author: Massimo Quadrana """


import numpy as np
import scipy.sparse as sps

from sklearn.linear_model import ElasticNet
from Base.BaseFunction import BaseFunction

import multiprocessing
from multiprocessing import Pool
from functools import partial



class SLIMElasticNetRecommender():

    def __init__(self, alpha=1e-4, l1_ratio=0.1, fit_intercept=False, copy_X=False, precompute=False,
                 selection='random',
                 max_iter=100, tol=1e-4, topK=100, positive_only=True, workers=multiprocessing.cpu_count(),
                 use_tail_boost=False):

        self.analyzed_items = 0
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.precompute = precompute
        self.selection = selection
        self.max_iter = max_iter
        self.tol = tol
        self.topK = topK
        self.positive_only = positive_only
        self.workers = workers
        self.helper = BaseFunction()

    def _partial_fit(self, currentItem, X):
        model = ElasticNet(alpha=self.alpha,
                            l1_ratio=self.l1_ratio,
                            positive=self.positive_only,
                            fit_intercept=self.fit_intercept,
                            copy_X=self.copy_X,
                            precompute=self.precompute,
                            selection=self.selection,
                            max_iter=self.max_iter,
                            tol=self.tol)

        X_j = X.copy()
        y = X_j[:, currentItem].toarray()
        X_j.data[X_j.indptr[currentItem]:X_j.indptr[currentItem + 1]] = 0.0

        model.fit(X_j, y)
        local_topK = min(len(model.sparse_coef_.data) - 1, self.topK)

        relevant_items_partition = (-model.coef_).argpartition(local_topK)[0:local_topK]
        relevant_items_partition_sorting = np.argsort(-model.coef_[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        not_zero_mask = model.coef_[ranking] > 0.0
        ranking = ranking[not_zero_mask]

        values = model.coef_[ranking]
        rows = ranking
        cols = [currentItem] * len(ranking)

        return values, rows, cols

    def fit(self, URM):

        self.URM_train = sps.csc_matrix(URM)

        n_items = self.URM_train.shape[1]
        print("Iterating for " + str(n_items) + " times")

        # create a copy of the URM since each _pfit will modify it
        copy_urm = self.URM_train.copy()

        _pfit = partial(self._partial_fit, X=self.URM_train, topK=self.topK)
        pool = Pool(processes=self.workers)
        res = pool.map(_pfit, np.arange(n_items))

        # res contains a vector of (values, rows, cols) tuples
        values, rows, cols = [], [], []
        for values_, rows_, cols_ in res:
            values.extend(values_)
            rows.extend(rows_)
            cols.extend(cols_)

        self.W_sparse = sps.csc_matrix((values, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def get_expected_ratings(self, user_id):
        user_profile = self.URM_train[user_id]
        expected_ratings = user_profile.dot(self.W_sparse).toarray().ravel()
        return expected_ratings

    def recommend(self, user_id, at=10):

        # compute the scores using the dot product
        scores = self.get_expected_ratings(user_id)
        user_profile = self.URM_train[user_id].indices
        scores[user_profile] = 0

        # rank items
        recommended_items = np.flip(np.argsort(scores), 0)

        return recommended_items[:at]