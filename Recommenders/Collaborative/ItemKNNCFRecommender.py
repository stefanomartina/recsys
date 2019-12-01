""" @author: Simone Lanzillotta, Stefano Martina """

from Utils.Compute_Similarity_Python import Compute_Similarity_Python, check_matrix
from Utils.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np

class ItemKNNCFRecommender(object):
    RECOMMENDER_NAME = "ItemKNNCFRecommender"

    def __init__(self):
        pass

    def feature_weight(self, feature_weighting):
        if feature_weighting == "BM25":
            self.URM = self.URM.astype(np.float32)
            self.URM = okapi_BM_25(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM = self.URM.astype(np.float32)
            self.URM = TF_IDF(self.URM.T).T
            self.URM = check_matrix(self.URM, 'csr')

    def compute_similarity(self, topK=10, shrink=50, normalize=True, similarity="tversky", feature_weighting="TF-IDF"):

        if not None:
            self.feature_weight(feature_weighting)

        similarity_object = Compute_Similarity_Python(self.URM,
                                                      shrink=shrink,
                                                      topK=topK,
                                                      normalize=normalize,
                                                      similarity=similarity)
        return similarity_object

    def fit(self, URM, ICM = None):
        self.URM = URM
        similarity_object = self.compute_similarity()

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=10, exclude_seen=True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        ranking = scores.argsort()[::-1]
        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores
