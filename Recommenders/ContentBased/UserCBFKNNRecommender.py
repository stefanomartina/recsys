""" @author: Simone Lanzillotta, Stefano Martina """

"""
    This class refers to a UserCBFKNNRecommender which uses a similarity matrix. In particular, can compute different 
    combination of multiple UCM inputs. In our case, we have 2 different sources from which generate an UCM_merged
"""

from Utils.Compute_Similarity_Python import Compute_Similarity_Python, check_matrix
from Utils.IR_feature_weighting import okapi_BM_25, TF_IDF
import scipy.sparse as sps
import numpy as np


class UserCBFKNNRecommender():
    RECOMMENDER_NAME = "UserCBFKNNRecommender("

    def fit(self, URM, list_UCM, topK=200, shrink=100, normalize=True,  similarity="tversky", feature_weighting="TF-IDF"):
        self.URM = URM
        self.UCM_age, self.UCM_region = list_UCM

        denseMatrix_region = self.UCM_region.todense()
        denseMatrix_age = self.UCM_age.todense()
        mergedMatrixDense = np.concatenate((denseMatrix_age, denseMatrix_region), axis = 1)
        self.UCM_merged = sps.csr_matrix(mergedMatrixDense)

        if feature_weighting == "BM25":
            self.URM = self.URM.astype(np.float32)
            self.URM = okapi_BM_25(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM = self.URM.astype(np.float32)
            self.URM = TF_IDF(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

        self.similarity = Compute_Similarity_Python(self.UCM_merged.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)

        self.W_sparse = self.similarity.compute_similarity()
        self.similarityProduct = self.W_sparse.dot(self.URM)

    def recommend(self, user_id, at=10, exclude_seen=True):

        scores = (self.similarityProduct[user_id]).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id+1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] -= np.inf

        return scores