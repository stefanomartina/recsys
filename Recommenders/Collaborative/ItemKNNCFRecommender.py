""" @author: Simone Lanzillotta, Stefano Martina """

from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction
import numpy as np

RECOMMENDER_NAME = "ItemKNNCFRecommender"

class ItemKNNCFRecommender():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, knn=500, shrink=100, similarity="tversky", normalize=True, feature_weighting=None):
        self.URM = URM
        # Compute similarity
        self.similarity_object = Compute_Similarity_Cython(self.URM, shrink=shrink, topK=knn, normalize=normalize, similarity=similarity)
        self.W_sparse = self.similarity_object.compute_similarity()
        self.similarityProduct = self.URM.dot(self.W_sparse)

    def get_expected_ratings(self, user_id):
        expected_scores = (self.similarityProduct[user_id]).toarray().ravel()
        return expected_scores

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def recommend(self, user_id, at=10, exclude_seen=True):

        expected_scores = self.get_expected_ratings(user_id)

        if exclude_seen:
            expected_scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]
        return ranking[:at]