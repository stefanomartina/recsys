from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np


item_cf_param = {
    "knn": 10,
    "shrink": 5,
}

user_cb_param = {
    "knn": 800,
    "shrink": 5,
}

slim_param = {
    "epochs": 200,
    "topK": 100,
}

class Hybrid_Combo3(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, list_ICM=None, list_UCM=None, weights=[0.2,0.2, 0.2],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercb=user_cb_param["knn"], shrink_usercb=user_cb_param["shrink"],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.list_ICM = list_ICM
        self.list_UCM = list_UCM
        self.rec_for_colder.fit(self.URM)
        self.cumulative_ifc_r = 0
        self.cumulative_slim_r = 0
        self.cumulative_ucbf_r = 0
        self.n_icf = 0
        self.n_slim = 0
        self.n_ucbf = 0
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

        if self.itemCF_ratings.sum(axis=0) != 0:
            self.n_icf += 1
            self.cumulative_ifc_r += self.itemCF_ratings.sum(axis=0)

        if self.userContentBased_ratings.sum(axis=0) != 0:
            self.n_ucbf += 1
            self.cumulative_ucbf_r += self.userContentBased_ratings.sum(axis=0)

        if self.slim_ratings.sum(axis=0) != 0:
            self.n_slim += 1
            self.cumulative_slim_r += self.slim_ratings.sum(axis=0)

    def sum_ratings(self):
        self.hybrid_ratings = self.slim_ratings * (self.weights[0])
        self.hybrid_ratings += self.userContentBased_ratings * (self.weights[1])
        self.hybrid_ratings += self.itemCF_ratings * (self.weights[2])

    def stats(self):
        print("self.cumulative_ifc_r/n")
        print(self.cumulative_ifc_r / self.n_icf)

        print("self.userContentBased_ratings/n")
        print(self.cumulative_ucbf_r/self.n_ucbf)
        print(self.n_ucbf)
        print(self.cumulative_ucbf_r)

        print("self.cumulative_slim_r/n")
        print(self.cumulative_slim_r / self.n_slim)