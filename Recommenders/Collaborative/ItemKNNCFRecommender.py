""" @author: Simone Lanzillotta, Stefano Martina """

from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction
import numpy as np
import scipy.sparse as sps
import os

RECOMMENDER_NAME = "ItemKNNCFRecommender"
SIMILARITY_PATH = "/SimilarityProduct/ItemCF_similarity.npz"

class ItemKNNCFRecommender():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, knn=700, shrink=5, similarity="tversky", normalize=True, transpose=False, tuning=False, feature_weighting=None):
        print("Fitting Item Collaborative Filtering Recommender...")
        self.URM = URM

        if feature_weighting is not None:
            self.helper.feature_weight(URM, feature_weighting)

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_hybrid(self.URM, SIMILARITY_PATH, knn, shrink,
                                                                         similarity, normalize, transpose=transpose,
                                                                         tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(self.URM, knn, shrink, similarity, normalize,
                                                                  transpose=transpose)
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