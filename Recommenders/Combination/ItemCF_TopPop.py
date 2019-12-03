""" @author: Simone Lanzillotta, Stefano Martina """
import numpy as np

from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender

class ItemCF_TopPop():


    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, knn=350, shrink=20, normalize=True, similarity="tversky"):
        self.URM = URM
        self.TP = TopPopRecommender()
        self.TP.fit(self.URM)
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

