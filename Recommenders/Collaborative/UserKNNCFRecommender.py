import numpy as np
from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
from Base.BaseFunction import BaseFunction

class UserKNNCFRecommender(object):

    def __init__(self):
        self.helper = BaseFunction()


    def fit(self, URM, knn=300, shrink=50, normalize=True, similarity="cosine"):
        self.URM = URM
        similarity_object = Compute_Similarity_Cython(URM.transpose(), shrink=shrink, topK=knn, normalize=normalize, similarity=similarity)
        self.SM_user = similarity_object.compute_similarity()
        self.RECS = self.SM_user.dot(self.URM)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.RECS[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]