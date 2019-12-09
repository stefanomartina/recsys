from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np


item_cf_param = {
    "knn": 10,
    "shrink": 5,
}


slim_param = {
    "epochs": 200,
    "topK": 10,
}

class Hybrid_Combo4(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, weights=[0.2, 0.2],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.rec_for_colder.fit(self.URM)

        # Sub-Fitting
        self.slim_random.fit(URM.copy())
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)


    def sum_ratings(self):
        self.hybrid_ratings = self.slim_ratings * (self.weights[0])
        self.hybrid_ratings += self.itemCF_ratings * (self.weights[1])

