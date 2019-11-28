from utils.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from recommenders import ItemCBFKNNRecommender

class HybridItemCF_ItemCB(object):
    # tversky
    # tanimoto
    def fit(self, URM, ICM, topK=10, shrink=50, normalize=True, similarity="tversky", threshold=0.1):
        self.URM = URM
        self.ICM_list = ICM
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()
        self.CB = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
        self.CB.fit(self.URM, self.ICM_list)
        self.threshold = threshold

    def recommend(self, user_id, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        summed_score = scores.sum(axis=0)
        # "Hybrid version with TOP-POP"
        if (summed_score <= self.threshold):
            return self.CB.recommend(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores
