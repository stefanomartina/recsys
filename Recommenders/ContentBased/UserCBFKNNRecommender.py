""" @author: Simone Lanzillotta, Stefano Martina """

from Utils.Compute_Similarity_Python import Compute_Similarity_Python
from Base.BaseFunction import BaseFunction
import scipy.sparse as sps
import numpy as np

RECOMMENDER_NAME = "UserCBFKNNRecommender("


class UserCBFKNNRecommender():

    def __init__(self, knn=300, shrink=4, similarity="tversky", normalize=True, feature_weighting="TF-IDF"):
        self.knn = knn
        self.shrink = shrink
        self.similarity = similarity
        self.normalize = normalize
        self.feature_weighting = feature_weighting
        self.helper = BaseFunction()
        self.URM = None

    def fit(self, URM, list_UCM):
        self.URM = URM
        self.UCM_age, self.UCM_region = list_UCM

        denseMatrix_region = self.UCM_region.todense()
        denseMatrix_age = self.UCM_age.todense()
        mergedMatrixDense = np.concatenate((denseMatrix_age, denseMatrix_region), axis = 1)
        self.UCM_merged = sps.csr_matrix(mergedMatrixDense)

        # IR Feature Weighting
        self.URM = self.helper.feature_weight(self.URM, self.feature_weighting)

        # Compute similarity
        self.similarity = Compute_Similarity_Python(self.UCM_merged.T, shrink=self.shrink, topK=self.knn, normalize=self.normalize, similarity=self.similarity)
        self.W_sparse = self.similarity.compute_similarity()
        self.similarityProduct = self.W_sparse.dot(self.URM)

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
            scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]

        return ranking[:at]

