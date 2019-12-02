""" @author: Simone Lanzillotta, Stefano Martina """

from Utils.Compute_Similarity_Python import Compute_Similarity_Python
from Base.BaseFunction import BaseFunction
import numpy as np

RECOMMENDER_NAME = "ItemKNNCFRecommender"

class ItemKNNCFRecommender():

    def __init__(self, knn=300, shrink=4, similarity="tversky", normalize=True, feature_weighting=None):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.normalize = normalize
        self.feature_weighting = feature_weighting
        self.helper = BaseFunction()
        self.URM = None

    def fit(self, URM):
        self.URM = URM
        # Compute similarity
        self.similarity_object = Compute_Similarity_Python(self.URM, shrink=self.shrink, topK=self.knn, normalize=self.normalize, similarity=self.similarity)
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