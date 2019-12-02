import pandas as pd

from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import UserCBFKNNRecommender, ItemCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.BaseFunction import BaseFunction
from Recommenders.MatrixFactorization.PureSVD.PureSVDRecommender import PureSVDRecommender

import numpy as np

user_cf_param = {
    "knn": 140,
    "shrink": 0
}

item_cf_param = {
    "knn": 310,
    "shrink": 0
}

cbf_param = {
    "knn": 45,
    "shrink": 8,
}

slim_param = {
    "epochs": 40,
    "topK": 200
}


class HybridRecommender(object):

    #######################################################################################
    #                                  INIT ALGORITHM                                     #
    #######################################################################################

    def __init__(self, user_cf_param = user_cf_param, item_cf_param = item_cf_param, cbf_param = cbf_param,
                 slim_param = slim_param):

        # User Content Based
        self.userContentBased = UserCBFKNNRecommender.UserCBFKNNRecommender(knn=user_cf_param["knn"], shrink=user_cf_param["shrink"])

        # Item Content Based
        self.itemContentBased = ItemCBFKNNRecommender.ItemCBFKNNRecommender(knn=item_cf_param["knn"], shrink=item_cf_param["shrink"])

        # Collaborative Filtring
        self.cbf = ItemKNNCFRecommender(knn=cbf_param["knn"], shrink=cbf_param["shrink"])

        # Slim
        self.slim_random = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"])

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, list_ICM, list_UCM):
        self.URM = URM

        # Sub-Fitting
        print("Fitting UserContentRecommender...")
        self.userContentBased.fit(URM.copy(), list_UCM)

        print("Fitting ItemContentRecommender...")
        self.itemContentBased.fit(URM.copy(), list_ICM)

        print("Fitting Collaborative Filtering...")
        self.cbf.fit(URM.copy())

        print("Fitting slim...")
        self.slim_random.fit(URM.copy())

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def recommend(self, user_id, weights=[0.25,0.25,0.25,0.25], at=10):

        self.hybrid_ratings = None

        self.userContentBased_ratings = self.userContentBased.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.cbf_ratings = self.cbf.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)

        self.hybrid_ratings = self.userContentBased_ratings * weights[0]
        self.hybrid_ratings += self.itemContentBased_ratings * weights[1]
        self.hybrid_ratings += self.cbf_ratings * weights[2]
        self.hybrid_ratings += self.slim_ratings * weights[3]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################


