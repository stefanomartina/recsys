import numpy as np
from utils.Compute_Similarity_Python import Compute_Similarity_Python


class ItemCBFKNNRecommender():

    def fit(self, URM, ICM, topK=50, shrink=100, normalize=True, similarity="cosine"):
        self.URM = URM
        self.ICM = ICM
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

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