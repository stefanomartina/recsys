import pandas as pd

from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import UserCBFKNNRecommender, ItemCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Base.BaseFunction import BaseFunction
from Recommenders.MatrixFactorization.PureSVD.PureSVDRecommender import PureSVDRecommender

import numpy as np

user_cf_param = {
    "knn": 800,
    "shrink": 100
}

item_cf_param = {
    "knn": 350,
    "shrink": 20
}

cf_param = {
    "knn": 10,
    "shrink": 25,
}

slim_param = {
    "epochs": 300,
    "topK": 10
}


class HybridRecommender(object):

    #######################################################################################
    #                                  INIT ALGORITHM                                     #
    #######################################################################################

    def __init__(self, slim_param = slim_param):

        # User Content Based
        self.userContentBased = UserCBFKNNRecommender.UserCBFKNNRecommender()

        # Item Content Based
        self.itemContentBased = ItemCBFKNNRecommender.ItemCBFKNNRecommender()

        # Collaborative Filtring
        self.cf = ItemKNNCFRecommender()

        # Slim
        self.slim_random = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"])

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, list_ICM, list_UCM, weights, knn_user=user_cf_param["knn"], shrink_user=user_cf_param["shrink"],
            knn_item=item_cf_param["knn"], shrink_item=item_cf_param["shrink"], knn_cf=cf_param["knn"], shrink_cf=cf_param["shrink"]):
        self.URM = URM
        self.weights = weights

        # Sub-Fitting
        print("Fitting UserContentRecommender...")
        self.userContentBased.fit(URM.copy(), list_UCM, knn_user, shrink_user)

        print("Fitting ItemContentRecommender...")
        self.itemContentBased.fit(URM.copy(), list_ICM, knn_item, shrink_item)

        print("Fitting Collaborative Filtering...")
        self.cf.fit(URM.copy(), knn_cf, shrink_cf)

        print("Fitting slim...")
        self.slim_random.fit(URM.copy())

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def recommend(self, user_id, at=10):

        self.hybrid_ratings = None

        self.userContentBased_ratings = self.userContentBased.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.cf_ratings = self.cf.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)

        self.hybrid_ratings = self.userContentBased_ratings * self.weights[0,0]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[0,1]
        self.hybrid_ratings += self.cf_ratings * self.weights[0,2]
        self.hybrid_ratings += self.slim_ratings * self.weights[0,3]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################


