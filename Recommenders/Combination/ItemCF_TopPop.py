""" @author: Simone Lanzillotta, Stefano Martina """

import numpy as np

from Base.BaseFunction import BaseFunction
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender

RECOMMENDER_NAME = "ItemCF_TopPopRecommender"
SIMILARITY_PATH = "/SimilarityProduct/ItemCF_TopPop_similarity.npz"

class ItemCF_TopPop():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, knn=500, shrink=100, similarity="tversky", normalize=True, transpose=False, tuning=False, feature_weighting="TF-IDF"):
        self.URM = URM
        self.TP = TopPopRecommender()
        self.TP.fit(self.URM)

        if feature_weighting is not None:
            self.helper.feature_weight(URM, feature_weighting)

        # Compute similarity
        self.W_sparse = self.helper.get_cosine_similarity(self.URM, SIMILARITY_PATH, knn, shrink, similarity, normalize, transpose=transpose, tuning=tuning)
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

    def recommend(self, user_id, at=10, exclude_seen=True, threshold=0.3):
        self.threshold = threshold

        expected_score = self.similarityProduct[user_id].toarray().ravel()
        summed_score = expected_score.sum(axis=0)

        if (summed_score <= self.threshold):
            return self.TP.recommend(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, expected_score)

        # rank items
        ranking = expected_score.argsort()[::-1]
        return ranking[:at]

