from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np



item_cf_param = {
    "knn": 10,
    "shrink": 30,
}

user_cb_param = {
    "knn": 800,
    "shrink": 5,
}

item_cb_param = {
    "knn": 5,
    "shrink": 100,
}

slim_param = {
    "epochs": 200,
    "topK": 10,
}

class Hybrid_Combo1(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[0.2,0.2,0.2,0.2],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercb=user_cb_param["knn"], shrink_usercb=user_cb_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM)


        # Sub-Fitting
        self.userContentBased.fit(URM.copy(), UCM_all, knn_usercb, shrink_usercb, tuning=tuning)
        self.itemContentBased.fit(URM.copy(), ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning)
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)
        self.slim_random.fit(URM.copy())


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.userContentBased_ratings = self.userContentBased.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.userContentBased_ratings * self.weights[0]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[1]
        self.hybrid_ratings += self.itemCF_ratings * self.weights[2]
        self.hybrid_ratings += self.slim_ratings * self.weights[3]
