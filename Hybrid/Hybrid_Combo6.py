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


class Hybrid_Combo6(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################
    # |  82       |  0.04825  |  2.28     |  0.07372  |  0.002197 |  0.6802   | --> PUBLIC: 0.03430
    # |  78       |  0.0482   |  2.407    |  0.0563   |  0.003071 |  0.674    | -- NON PROVATO
    # |  91       |  0.04822  |  2.475    |  0.07787  |  0.001966 |  0.5208   | --> PUBLIC: 0.03419
    # |  154      |  0.04844  |  2.65     |  0.1702   |  0.002764 |  0.7887   | --> NON PROVATO Locale = 0.04844
    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[2.28,0.07372,0.002197,0.6802],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=item_cf_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all, tuning=tuning, similarity_path="/SimilarityProduct/UserCBFCold_similarity6.npz")

        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF_similarity6.npz")
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning, similarity_path="/SimilarityProduct/UserCF_similarity6.npz")
        self.itemContentBased.fit(URM.copy(), ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning, similarity_path="/SimilarityProduct/UserCB_similarity6.npz")
        self.elasticNet.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/ElasticNet_similarity6.npz")

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.elasticNet_ratings = self.elasticNet.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * self.weights[0]
        self.hybrid_ratings += self.userCF_ratings * self.weights[1]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[2]
        self.hybrid_ratings += self.elasticNet_ratings * self.weights[3]

    def extract_rating_hybrid(self):
        return self.hybrid_ratings