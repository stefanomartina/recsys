from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np


user_cf_param = {
    "knn": 600,
    "shrink": 0,
}

item_cf_param = {
    "knn": 10,
    "shrink": 30,
}

item_cb_param = {
    "knn": 5,
    "shrink": 100,
}

class Hybrid_Combo9(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[1.571,0.009125,0.004897,0.086],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=user_cf_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all)

        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF_similarity9.npz")
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning, similarity_path="/SimilarityProduct/UserCF_similarity9.npz")
        self.RP3Beta.fit(URM.copy())
        self.itemContentBased.fit(URM.copy(), self.ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning, similarity_path="/SimilarityProduct/ItemCB_similarity9.npz")

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.RP3Beta_ratings = self.RP3Beta.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * self.weights[0]
        self.hybrid_ratings += self.userCF_ratings * self.weights[1]
        self.hybrid_ratings += self.RP3Beta_ratings * self.weights[2]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[3]