"""
@author: Simone Lanzillotta, Stefano Martina
This class represents the general structure of some recommendation algorithm developed within this repository
"""

import numpy as np
from Base.BaseFunction import BaseFunction

RECOMMENDER_NAME = "AbstractBaseRecommender"

class BaseRecommender():

    def __init__(self):
        self.helper = BaseFunction()
        self.similarityProduct = None
        self.URM = None
        self.UserCBF = None

    def get_expected_ratings(self, user_id):
        expected_scores = (self.similarityProduct[user_id]).toarray().ravel()
        return expected_scores

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_scores = self.get_expected_ratings(user_id)
        summed_score = expected_scores.sum(axis=0)

        if exclude_seen:
            expected_scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]
        return ranking[:at]