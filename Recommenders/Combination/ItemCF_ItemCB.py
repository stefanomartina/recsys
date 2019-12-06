""" @author: Simone Lanzillotta, Stefano Martina """

import numpy as np
from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender

RECOMMENDER_NAME = "ItemCF_ItemCBRecommender"
SIMILARITY_PATH = "/SimilarityProduct/ItemCF_ItemCB_similarity.npz"

class ItemCF_ItemCB():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, ICM, knn=500, shrink=100, similarity="tversky", normalize=True, transpose=False, tuning=False, feature_weighting="TF-IDF"):
        self.URM = URM
        self.ICM_list = ICM
        self.CB = ItemCBFKNNRecommender()
        self.CB.fit(self.URM, self.ICM_list, tuning=tuning)

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

    def recommend(self, user_id, at=10, exclude_seen=True, threshold = 0.0):
        self.threshold = threshold

        expected_scores = self.get_expected_ratings(user_id)

        summed_score = expected_scores.sum(axis=0)
        if (summed_score <= self.threshold):
            return self.CB.recommend(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, expected_scores)

        ranking = expected_scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores
