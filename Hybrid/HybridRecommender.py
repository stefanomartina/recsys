import pandas as pd

from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import UserCBFKNNRecommender, ItemCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Combination.ItemCF_ItemCB import ItemCF_ItemCB
from Recommenders.Combination.ItemCF_TopPop import ItemCF_TopPop
from Base.BaseFunction import BaseFunction
from Recommenders.MatrixFactorization.PureSVD.PureSVDRecommender import PureSVDRecommender

import numpy as np

user_cf_param = {
    "knn": 557,
    "shrink": 19
}

item_cf_param = {
    "knn": 147,
    "shrink": 36,
}

user_cb_param = {
    "knn": 800,
    "shrink": 5,
}

item_cb_param = {
    "knn": 700,
    "shrink": 5,
}

cf_param = {
    "knn": 5,
    "shrink": 10,
}

slim_param = {
    "epochs": 500,
    "topK": 200,
}

cftp_param = {
    "knn": 350,
    "shrink": 25,
}

cfcb_param = {
    "knn": 50,
    "shrink": 25,
}




class HybridRecommender(object):

    #######################################################################################
    #                                  INIT ALGORITHM                                     #
    #######################################################################################

    def __init__(self):

        self.hybrid_ratings = None

        # User Content Based
        self.userContentBased = UserCBFKNNRecommender.UserCBFKNNRecommender()

        # Item Content Based
        self.itemContentBased = ItemCBFKNNRecommender.ItemCBFKNNRecommender()

        # Item Collaborative + TopPopRecommender
        self.itemCF_itemCB_Combo = ItemCF_ItemCB()

        # Item Collaborative + Item Content Based
        self.itemCF_TopPop_Combo = ItemCF_TopPop()

        # Collaborative Filtring
        self.cf = ItemKNNCFRecommender()

        # Slim
        self.slim_random = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"])

        # Ratings from each available algorith
        self.userContentBased_ratings = None
        self.itemContentBased_ratings = None
        self.cf_ratings = None
        self.cf_tp_combo_ratings = None
        self.cf_cb_combo_ratings = None
        self.slim_ratings = None

    def switch_ratings(self, argument):
        switcher = {
            "UserContentBased": self.userContentBased_ratings,
            "ItemContentBased": self.itemContentBased_ratings,
            "ItemCF": self.cf_ratings,
            "ItemCF_TopPop_Combo": self.cf_tp_combo_ratings,
            "ItemCF_ItemCB_Combo": self.cf_cb_combo_ratings,
            "Slim": self.slim_ratings,
        }
        return switcher.get(argument, "Invalid argument")

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    # To force a production from hybrid recommender, set manually weights and move it after list_ICM and list_UCM

    def fit(self, URM, weights, list_ICM = None, list_UCM = None,
                   knn_user=user_cf_param["knn"], shrink_user=user_cf_param["shrink"],
                   knn_item=item_cf_param["knn"], shrink_item=item_cf_param["shrink"],
                   knn_cf=cf_param["knn"], shrink_cf=cf_param["shrink"],
                   knn_cftp=cftp_param["knn"], shrink_cftp=cftp_param["shrink"],
                   knn_cfcb=cfcb_param["knn"], shrink_cfcb=cfcb_param["shrink"], tuning=False, combination="third"):

        self.URM = URM
        self.weights = np.array(weights)

        # Sub-Fitting
        if combination == "first":
            self.userContentBased.fit(URM.copy(), list_UCM, knn_user, shrink_user, tuning=tuning)
            self.itemContentBased.fit(URM.copy(), list_ICM, knn_item, shrink_item, tuning=tuning)
            self.cf.fit(URM.copy(), knn_cf, shrink_cf, tuning=tuning)
            self.slim_random.fit(URM.copy())

        if combination == "second":
            self.cf.fit(URM.copy(), knn_cf, shrink_cf, tuning=tuning)
            self.slim_random.fit(URM.copy())
            self.itemCF_TopPop_Combo.fit(URM.copy(), knn_cftp, shrink_cftp, tuning=tuning)
            self.itemCF_itemCB_Combo.fit(URM.copy(), list_ICM, knn_cfcb, shrink_cfcb, tuning=tuning)

        if combination == "third":
            self.userContentBased.fit(URM.copy(), list_UCM, knn_user, shrink_user, tuning=tuning)
            self.slim_random.fit(URM.copy())
            self.cf.fit(URM.copy(), knn_cf, shrink_cf, tuning=tuning)

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def sum_score(self, user_id):
        self.userContentBased_ratings = self.userContentBased.get_expected_ratings(user_id)
        #self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.cf_ratings = self.cf.get_expected_ratings(user_id)
        #self.cf_tp_combo_ratings = self.itemCF_TopPop_Combo.get_expected_ratings(user_id)
        #self.cf_cb_combo_ratings = self.itemCF_itemCB_Combo.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)

    #######################################################################################
    #                                    RECOMMENDING                                     #
    #######################################################################################

    def recommend(self, user_id, at=10, hybrid="third"):

        self.sum_score(user_id)

        if hybrid == "first":
            self.hybrid_ratings = self.switch_ratings("UserContentBased") * self.weights[0]
            self.hybrid_ratings += self.switch_ratings("ItemContentbased") * self.weights[1]
            self.hybrid_ratings += self.switch_ratings("ItemCF") * self.weights[2]
            self.hybrid_ratings += self.switch_ratings("Slim") * self.weights[3]

        if hybrid == "second":
            self.hybrid_ratings = self.switch_ratings("ItemCF") * self.weights[0]
            self.hybrid_ratings += self.switch_ratings("Slim") * self.weights[1]
            self.hybrid_ratings += self.switch_ratings("ItemCF_TopPop_Combo") * self.weights[2]
            self.hybrid_ratings += self.switch_ratings("ItemCF_ItemCB_Combo") * self.weights[3]


        if hybrid == "third":
            self.hybrid_ratings = self.switch_ratings("Slim") * self.weights[0]
            self.hybrid_ratings += self.switch_ratings("UserContentBased") * (self.weights[1])
            self.hybrid_ratings += self.switch_ratings("ItemCF") * self.weights[2]

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)

        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]





