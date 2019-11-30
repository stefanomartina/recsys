import numpy as np
import pandas as pd

from utils.Compute_Similarity_Python import Compute_Similarity_Python, check_matrix
import scipy.sparse as sps

class ItemCBFKNNRecommender():

    """
        This class refers to a ItemCBFKNNRecommender which uses a similarity matrix. In particular, can compute
        different combination of multiple ICM inputs. In our case, we have 3 different sources from which generate
        ICM, so:
            - [First, Second]
            - [First, Third]
            - [Third, Second]
            - [First, Second, Third]
    """

    def fit(self, URM, list_ICM, topK=10, shrink=50, normalize=True, similarity="tversky"):

        # extract all relevant matrix from source
        self.URM = URM
        self.ICM = list_ICM[0]
        self.ICM_asset = list_ICM[1]
        self.ICM_price = list_ICM[2]

        # compute similarity_object
        self.ICM_merged = sps.hstack((self.ICM, self.ICM_asset, self.ICM_price)).tocsr()
        self.similarity = Compute_Similarity_Python(self.ICM_merged.T, shrink=shrink,
                                                  topK=topK, normalize=normalize,
                                                  similarity = similarity)

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