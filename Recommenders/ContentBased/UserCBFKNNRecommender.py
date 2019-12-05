""" @author: Simone Lanzillotta, Stefano Martina """

from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction
import scipy.sparse as sps
import numpy as np

RECOMMENDER_NAME = "UserCBFKNNRecommender"
SIMILARITY_PATH = "/SimilarityProduct/UserCB_similarity.npz"

class UserCBFKNNRecommender():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, list_UCM, knn=800, shrink=5, similarity="tversky", normalize=True, transpose=True, tuning=False, feature_weighting="TF_IDF"):
        print("Fitting User Content Based Recommender Recommender...")
        self.URM = URM
        self.UCM_age, self.UCM_region = list_UCM

        denseMatrix_region = self.UCM_region.todense()
        denseMatrix_age = self.UCM_age.todense()

        mergedMatrixDense = np.concatenate((denseMatrix_age, denseMatrix_region), axis=1)
        self.UCM_merged = sps.csr_matrix(mergedMatrixDense)

        # IR Feature Weighting
        if feature_weighting is not None:
            self.UCM_merged = self.helper.feature_weight(self.UCM_merged, feature_weighting)

        # Compute similarity
        self.W_sparse = self.helper.get_cosine_similarity(self.UCM_merged, SIMILARITY_PATH, knn, shrink, similarity, normalize, transpose=transpose, tuning=tuning)
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

