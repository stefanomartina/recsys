import pandas as pd

from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import UserCBFKNNRecommender, ItemCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Combination.ItemCF_ItemCB import ItemCF_ItemCB
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender

import numpy as np


user_cf_param = {
    "knn": 646,
    "shrink": 2
}

item_cf_param = {
    "knn": 6,
    "shrink": 42,
}

user_cb_param = {
    "knn": 900,
    "shrink": 0,
}

item_cb_param = {
    "knn": 700,
    "shrink": 5,
}

slim_param = {
    "epochs": 250,
    "topK": 100,
}

cftp_param = {
    "knn": 350,
    "shrink": 25,
}

cfcb_param = {
    "knn": 50,
    "shrink": 25,
}

COMBO6 = [0.4, 0.005, 1.5]


class HybridRecommender(object):

    #######################################################################################
    #                                  INIT ALGORITHM                                     #
    #######################################################################################

    def __init__(self, combination):

        self.hybrid_ratings = None
        self.combination = combination
        self.TP = TopPopRecommender()

        # User Content Based
        self.userContentBased = UserCBFKNNRecommender.UserCBFKNNRecommender()

        # Item Content Based
        self.itemContentBased = ItemCBFKNNRecommender.ItemCBFKNNRecommender()

        # Item Collaborative + Item Content Based
        self.itemCF_itemCB_Combo = ItemCF_ItemCB()

        # Item Collaborative + TopPopRecommender
        self.itemCF_TopPop_Combo = ItemCF_TopPop()

        # UserCollaborative + TopPopRecommender
        self.userCF_TopPop_Combo = UserCF_TopPop()

        # User Collaborative
        self.userCF = UserKNNCFRecommender()

        # Item Collaborative
        self.itemCF = ItemKNNCFRecommender()

        # Slim
        self.slim_random = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"])

        # Ratings from each available algorith
        self.userContentBased_ratings = None
        self.itemContentBased_ratings = None
        self.itemCF_ratings = None
        self.userCF_ratings = None
        self.icf_tp_combo_ratings = None
        self.ucf_tp_combo_ratings = None
        self.cf_cb_combo_ratings = None
        self.slim_ratings = None

    def switch_ratings(self, argument):
        switcher = {
            "UserContentBased": self.userContentBased_ratings,
            "ItemContentBased": self.itemContentBased_ratings,
            "ItemCF": self.itemCF_ratings,
            "UserCF": self.userCF_ratings,
            "ItemCF_TopPop_Combo": self.icf_tp_combo_ratings,
            "UserCF_TopPop_Combo": self.ucf_tp_combo_ratings,
            "ItemCF_ItemCB_Combo": self.cf_cb_combo_ratings,
            "Slim": self.slim_ratings,
        }
        return switcher.get(argument, "Invalid argument")

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    # To force a production from hybrid recommender, set manually weights and move it after list_ICM and list_UCM

    def fit(self, URM, list_ICM = None, list_UCM = None, weights=[0.01464, 0.992],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=user_cf_param["shrink"],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercb=user_cb_param["knn"], shrink_usercb=user_cb_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],
                   knn_cftp=cftp_param["knn"], shrink_cftp=cftp_param["shrink"],
                   knn_cfcb=cfcb_param["knn"], shrink_cfcb=cfcb_param["shrink"], tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.TP.fit(self.URM)

        # Sub-Fitting
        if self.combination == "Combo1":
            self.userContentBased.fit(URM.copy(), list_UCM, knn_usercb, shrink_usercb, tuning=tuning)
            self.itemContentBased.fit(URM.copy(), list_ICM, knn_itemcb, shrink_itemcb, tuning=tuning)
            self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)
            self.slim_random.fit(URM.copy())

        if self.combination == "Combo2":
            self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)
            self.slim_random.fit(URM.copy())
            self.itemCF_TopPop_Combo.fit(URM.copy(), knn_cftp, shrink_cftp, tuning=tuning)
            self.itemCF_itemCB_Combo.fit(URM.copy(), list_ICM, knn_cfcb, shrink_cfcb, tuning=tuning)

        if self.combination == "Combo3":
            self.userContentBased.fit(URM.copy(), list_UCM, knn_usercb, shrink_usercb, tuning=tuning, transpose=True)
            self.slim_random.fit(URM.copy())
            self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)

        if self.combination == "Combo4":
            self.userContentBased.fit(URM.copy(), list_UCM, knn_usercb, shrink_usercb, tuning=tuning, similarity_path="/SimilarityProduct/UserCB2_similarity.npz")
            self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF2_similarity.npz")

        if self.combination == "Combo5":
            self.userCF_TopPop_Combo.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning, similarity_path="/SimilarityProduct/UserCF-TopPop2_similarity.npz")
            self.itemCF_TopPop_Combo.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF-TopPop2_similarity.npz")

        if self.combination == "Combo6":
            self.userContentBased.fit(URM.copy(), list_UCM, knn_usercb, shrink_usercb, tuning=tuning, transpose=True, similarity_path="/SimilarityProduct/UserCB6_similarity.npz")
            self.slim_random.fit(URM.copy())
            self.itemCF_TopPop_Combo.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF-TopPop6_similarity.npz")

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def sum_score(self, user_id):
        self.userContentBased_ratings = self.userContentBased.get_expected_ratings(user_id)

        #self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        #self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        #self.userCF_ratings = self.userCF.get_expected_ratings(user_id)

        self.icf_tp_combo_ratings = self.itemCF_TopPop_Combo.get_expected_ratings(user_id)

        #self.ucf_tp_combo_ratings = self.userCF_TopPop_Combo.get_expected_ratings(user_id)

        #self.cf_cb_combo_ratings = self.itemCF_itemCB_Combo.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)


    #######################################################################################
    #                                    RECOMMENDING                                     #
    #######################################################################################

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores


    def recommend(self, user_id, at=10, exclude_seen=True):

        self.sum_score(user_id)

        if self.combination == "Combo1":
            self.hybrid_ratings = self.switch_ratings("UserContentBased") * self.weights[0]
            self.hybrid_ratings += self.switch_ratings("ItemContentbased") * self.weights[1]
            self.hybrid_ratings += self.switch_ratings("ItemCF") * self.weights[2]
            self.hybrid_ratings += self.switch_ratings("Slim") * self.weights[3]

        if self.combination == "Combo2":
            self.hybrid_ratings = self.switch_ratings("ItemCF") * self.weights[0]
            self.hybrid_ratings += self.switch_ratings("Slim") * self.weights[1]
            self.hybrid_ratings += self.switch_ratings("ItemCF_TopPop_Combo") * self.weights[2]
            self.hybrid_ratings += self.switch_ratings("ItemCF_ItemCB_Combo") * self.weights[3]


        if self.combination == "Combo3":
            self.hybrid_ratings = self.switch_ratings("Slim") * (self.weights[0])
            self.hybrid_ratings += self.switch_ratings("UserContentBased") * (self.weights[1])
            self.hybrid_ratings += self.switch_ratings("ItemCF") * (self.weights[2])

        if self.combination == "Combo4":
            self.hybrid_ratings = self.switch_ratings("UserContentBased") * (self.weights[0])
            self.hybrid_ratings += self.switch_ratings("ItemCF") * (self.weights[1])


        if self.combination == "Combo5":
            self.hybrid_ratings = self.switch_ratings("UserCF_TopPop_Combo") * (self.weights[0])
            self.hybrid_ratings += self.switch_ratings("ItemCF_TopPop_Combo") * (self.weights[1])

        if self.combination == "Combo6":
            self.hybrid_ratings = self.switch_ratings("Slim") * (self.weights[0])
            self.hybrid_ratings += self.switch_ratings("UserContentBased") * (self.weights[1])
            self.hybrid_ratings += self.switch_ratings("ItemCF_TopPop_Combo") * (self.weights[2])

        summed_score = self.hybrid_ratings.sum(axis=0)

        if (summed_score == 0):
            return self.TP.recommend(user_id)

        else:
            recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)
            # REMOVING SEEN
            unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

            return recommended_items[0:at]





