""" @author: Simone Lanzillotta, Stefano Martina """

import numpy as np
from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender

class ItemCF_ItemCB():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, ICM, knn=10, shrink=50, normalize=True, similarity="tversky", feature_weighting="TF-IDF"):
        self.URM = URM
        self.ICM_list = ICM
        self.CB = ItemCBFKNNRecommender()
        self.CB.fit(self.URM, self.ICM_list)

        if feature_weighting is not None:
            self.helper.feature_weight(URM, feature_weighting)

        similarity_object = Compute_Similarity_Cython(URM, shrink=shrink, topK=knn, normalize=normalize, similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()
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
