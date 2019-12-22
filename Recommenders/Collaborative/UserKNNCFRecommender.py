import numpy as np
from Base.BaseFunction import BaseFunction

RECOMMENDER_NAME = "UserKNNCFRecommender"
SIMILARITY_PATH = "/SimilarityProduct/UserCF_similarity.npz"

""" Working with URM.transpose() - Transpose field = True """

class UserKNNCFRecommender(object):

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, knn=600, shrink=0, similarity="cosine", normalize=True, transpose=True, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_stored(self.URM, RECOMMENDER_NAME, similarity_path, knn, shrink,
                                                                         similarity, normalize, transpose=transpose,
                                                                         tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(self.URM, knn, shrink, similarity, normalize,
                                                                  transpose=transpose)

        self.similarityProduct = self.W_sparse.dot(self.URM)


    def get_expected_ratings(self, user_id):
        expected_ratings = self.similarityProduct[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]