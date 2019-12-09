from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np


item_cf_param = {
    "knn": 6,
    "shrink": 42,
}

user_cb_param = {
    "knn": 900,
    "shrink": 0,
}

slim_param = {
    "epochs": 250,
    "topK": 100,
}

class Hybrid_Combo3(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, list_ICM=None, list_UCM=None, weights=[0.2,0.2],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercb=user_cb_param["knn"], shrink_usercb=user_cb_param["shrink"],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.list_ICM = list_ICM
        self.list_UCM = list_UCM
        self.rec_for_colder.fit(self.URM)

        # Sub-Fitting
        self.userContentBased.fit(URM.copy(), list_UCM, knn_usercb, shrink_usercb, tuning=tuning, transpose=True)
        self.slim_random.fit(URM.copy())
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userContentBased_ratings = self.userContentBased.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.slim_ratings * (self.weights[0])
        self.hybrid_ratings += self.userContentBased_ratings * (self.weights[1])
        self.hybrid_ratings += self.itemCF_ratings * (self.weights[2])