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


class Hybrid_Combo5(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[0.2,0.2],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=user_cf_param["shrink"],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM)


        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning)

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.userCF_ratings * self.weights[0]
        self.hybrid_ratings += self.itemCF_ratings * self.weights[1]