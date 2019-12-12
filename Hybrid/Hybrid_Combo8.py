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



class Hybrid_Combo8(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[1.571,0.009125,0.004897,0.086],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=item_cf_param["shrink"],tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM)

        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF_similarity8.npz")
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning,similarity_path="/SimilarityProduct/UserCF_similarity8.npz")
        self.pureSVD.fit(URM.copy())
        self.elasticNet.fit(URM.copy(), tuning=tuning)

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.pureSVD_ratings = self.pureSVD.get_expected_ratings(user_id)
        self.elasticNet_ratings = self.elasticNet.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * self.weights[0]
        self.hybrid_ratings += self.userCF_ratings * self.weights[1]
        self.hybrid_ratings += self.pureSVD_ratings * self.weights[2]
        self.hybrid_ratings += self.elasticNet_ratings * self.weights[3]