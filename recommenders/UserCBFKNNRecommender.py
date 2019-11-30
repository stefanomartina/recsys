import numpy as np
import pandas as pd

from utils.Compute_Similarity_Python import Compute_Similarity_Python, check_matrix
import scipy.sparse as sps

class UserCBFKNNRecommender():

    """
        This class refers to a UserCBFKNNRecommender which uses a similarity matrix. In particular, can compute
        different combination of multiple ICM inputs. In our case, we have 3 different sources from which generate
        ICM, so:

    """

    def fit(self, URM, list_UCM, topK=10, shrink=50, normalize=True, similarity="tversky"):

        self.URM = URM
        # extract all relevant matrix from source
        self.UCM_age = list_UCM[0]
        self.UCM_region = list_UCM[1]

        denseMatrix_region = self.UCM_region.todense()
        denseMatrix_age = self.UCM_age.todense()

        mergedMatrixDense = np.concatenate((denseMatrix_age, denseMatrix_region), axis = 1)

        self.UCM_merged = sps.csr_matrix(mergedMatrixDense)

        # compute similarity_object
        self.similarity = Compute_Similarity_Python(self.UCM_merged.T, shrink=shrink,
                                                  topK=topK, normalize=normalize,
                                                  similarity = similarity)

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