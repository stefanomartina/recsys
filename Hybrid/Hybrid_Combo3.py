from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np



item_cf_param = {
    "knn": 10,
    "shrink": 30,
}

user_cf_param = {
    "knn": 600,
    "shrink": 0,
}

item_cb_param = {
    "knn": 5,
    "shrink": 0,
}

rp3beta_param = {
    "alpha":0.3515,
    "beta":0.1003,
    "topK":90,
}


class Hybrid_Combo3(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################
    # |  6        |  0.0493   |  2.895    |  0.08504  |  0.03905  |  0.6359   |  2.634    |
    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[2.895, 0.08504, 0.03905, 2.634],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=item_cf_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],
                   alpha_rp3beta=rp3beta_param["alpha"], beta_rp3beta=rp3beta_param["beta"], topk_rp3beta=rp3beta_param["topK"],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all, tuning=tuning)

        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning)
        self.itemContentBased.fit(URM.copy(), ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning)
        self.RP3Beta.fit(URM.copy())


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.RP3Beta_ratings = self.RP3Beta.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * self.weights[0]
        self.hybrid_ratings += self.userCF_ratings * self.weights[1]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[2]
        self.hybrid_ratings += self.RP3Beta_ratings * self.weights[3]