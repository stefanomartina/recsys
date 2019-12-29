""" @author: Simone Lanzillotta, Stefano Martina """

from Base.BaseFunction import BaseFunction
import scipy.sparse as sps
import numpy as np

RECOMMENDER_NAME = "ItemCBFKNNRecommender"
SIMILARITY_PATH = "/SimilarityProduct/ItemCB_similarity.npz"

class ItemCBFKNNRecommender():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, ICM_all, knn=5, shrink=100, similarity="cosine", normalize=True, transpose=True, feature_weighting = None, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM
        self.ICM_all = ICM_all

        if feature_weighting is not None:
            self.ICM_all = self.helper.feature_weight(self.ICM_all, feature_weighting)

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_stored(sps.hstack((self.ICM_all, self.URM.T)), RECOMMENDER_NAME, similarity_path, knn, shrink, similarity, normalize, transpose=transpose, tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(self.ICM_all, knn, shrink, similarity, normalize, transpose=transpose)

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
