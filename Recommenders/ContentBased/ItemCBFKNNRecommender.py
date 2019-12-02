""" @author: Simone Lanzillotta, Stefano Martina """

from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction
import scipy.sparse as sps
import numpy as np

RECOMMENDER_NAME = "ItemCBFKNNRecommender"


class ItemCBFKNNRecommender():

    def __init__(self):
        self.helper = BaseFunction()


    def fit(self, URM, list_ICM, knn=200, shrink=50, similarity="tversky", normalize=True, feature_weighting=None):
        self.URM = URM
        self.ICM, self.ICM_asset, self.ICM_price = list_ICM

        denseMatrix_ICM = self.ICM.todense()
        denseMatrix_ICM_asset = self.ICM_asset.todense()
        denseMatrix_ICM_price = self.ICM_price.todense()

        mergedMatrixDense = np.concatenate((denseMatrix_ICM, denseMatrix_ICM_asset, denseMatrix_ICM_price), axis=1)
        self.ICM_merged = sps.csr_matrix(mergedMatrixDense)

        # IR Feature Weighting
        self.ICM_merged = self.helper.feature_weight(self.ICM_merged, feature_weighting)

        # Compute similarity
        self.similarity = Compute_Similarity_Cython(self.ICM_merged.T, shrink=shrink, topK=knn, normalize=normalize, similarity=similarity)
        self.W_sparse = self.similarity.compute_similarity()
        self.similarityProduct = self.URM.dot(self.W_sparse)

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] -= np.inf

        return scores

    def get_expected_ratings(self, user_id):
        expected_scores = (self.similarityProduct[user_id]).toarray().ravel()
        return expected_scores

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_scores = self.get_expected_ratings(user_id)

        if exclude_seen:
            expected_scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]

        return ranking[:at]
