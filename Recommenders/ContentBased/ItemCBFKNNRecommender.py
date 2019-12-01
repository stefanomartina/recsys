""" @author: Simone Lanzillotta, Stefano Martina """

"""
    This class refers to a ItemCBFKNNRecommender which uses a similarity matrix. In particular, can compute different 
    combination of multiple ICM inputs. In our case, we have 2 different sources from which generate an ICM_merged
"""


from Utils.Compute_Similarity_Python import Compute_Similarity_Python, check_matrix
from Utils.IR_feature_weighting import okapi_BM_25, TF_IDF
import scipy.sparse as sps
import numpy as np

class ItemCBFKNNRecommender():
    RECOMMENDER_NAME = "ItemCBFKNNRecommender"

    def fit(self, URM, list_ICM, topK=10, shrink=50, normalize=True, similarity="tversky", feature_weighting=None):

        self.URM = URM
        self.ICM, self.ICM_asset, self.ICM_price = list_ICM
        self.ICM_merged = sps.hstack((self.ICM, self.ICM_asset, self.ICM_price)).tocsr()

        if feature_weighting == "BM25":
            self.ICM_merged = self.ICM_merged.astype(np.float32)
            self.ICM_merged = okapi_BM_25(self.ICM_merged)

        elif feature_weighting == "TF-IDF":
            self.ICM_merged = self.ICM_merged.astype(np.float32)
            self.ICM_merged = TF_IDF(self.ICM_merged)

        self.similarity = Compute_Similarity_Python(self.ICM_merged.T, shrink=shrink,
                                                  topK=topK, normalize=normalize,
                                                  similarity=similarity)

        self.W_sparse = self.similarity.compute_similarity()

    def recommend(self, user_id, at=10, exclude_seen=True):
        user_profile = self.URM[user_id]
        scores = (user_profile.dot(self.W_sparse)).toarray().ravel()

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