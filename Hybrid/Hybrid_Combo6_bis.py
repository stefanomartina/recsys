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


class Hybrid_Combo6_bis(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################
    # |  6        |  0.0493   |  2.895    |  0.08504  |  0.03905  |  0.6359   |  2.634    |
    # |  23       |  0.04933  |  2.844    |  0.08422  |  0.0414   |  0.6129   |  2.644    |
    # |  6        |  0.04956  |  2.689    |  0.2085   |  0.1039   |  0.467    |  2.688    |

    '''

    Hybrid6_bis MAP : 0.04933  |  2.844    |  0.08422  |  0.0414   |  0.6129   |  2.644
    '''

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[2.844 ,0.08422,0.0414,0.6129 , 2.644, 5],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=item_cf_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],
                   alpha_rp3beta=rp3beta_param["alpha"], beta_rp3beta=rp3beta_param["beta"], topk_rp3beta=rp3beta_param["topK"],
                   tuning=False):

        self.TopPop.fit(URM)
        self.UserCBF.fit(URM, UCM_all)
        self.get_cold_users(URM)
        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all)

        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF_similarity6.npz")
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning, similarity_path="/SimilarityProduct/UserCF_similarity6.npz")
        self.itemContentBased.fit(URM.copy(), ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning, similarity_path="/SimilarityProduct/ItemCB_similarity6.npz")
        self.elasticNet.fit(URM.copy(), tuning=tuning)
        self.RP3Beta.fit(URM.copy())
        self.slim_random.fit(URM.copy())


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.elasticNet_ratings = self.elasticNet.get_expected_ratings(user_id)
        self.RP3Beta_ratings = self.RP3Beta.get_expected_ratings(user_id)
        self.slim_ratings = self.RP3Beta.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * self.weights[0]
        self.hybrid_ratings += self.userCF_ratings * self.weights[1]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[2]
        self.hybrid_ratings += self.elasticNet_ratings * self.weights[3]
        self.hybrid_ratings += self.RP3Beta_ratings * self.weights[4]
        self.hybrid_ratings += self.slim_ratings * self.weights[5]

    def extract_rating_hybrid(self):
        return self.hybrid_ratings