import numpy as np
from Base.BaseFunction import BaseFunction
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender

RECOMMENDER_NAME = "UserCF_TopPopRecommender"
SIMILARITY_PATH = "/SimilarityProduct/UserCF_TopPop_similarity.npz"

""" Working with URM.transpose() - Transpose field = True """

class UserCF_TopPop(object):

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, knn=646, shrink=2, similarity="tversky", normalize=True, transpose=True, tuning=False, similarity_path=SIMILARITY_PATH):
        print("Fitting User Collaborative Filerting Recommender...")
        self.URM = URM

        self.TP = TopPopRecommender()
        self.TP.fit(self.URM)

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_hybrid(self.URM, similarity_path, knn, shrink,
                                                                         similarity, normalize, transpose=transpose,
                                                                         tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(self.URM, knn, shrink, similarity, normalize,
                                                                  transpose=transpose)

        self.similarityProduct = self.W_sparse.dot(self.URM)

    def get_expected_ratings(self, user_id):
        expected_ratings = self.similarityProduct[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def recommend(self, user_id, at=10,exclude_seen=True):
        expected_scores = self.get_expected_ratings(user_id)

        summed_score = expected_scores.sum(axis=0)

        if (summed_score == 0):
            return self.TP.recommend(user_id)

        if exclude_seen:
            expected_scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]

        return ranking[0:at]