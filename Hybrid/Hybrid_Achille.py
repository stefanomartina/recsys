""" @author: Simone Lanzillotta, Stefano Martina """

from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np

item_cf_param = {
    "knn": 63.62,
    "shrink": 24.14,
}

user_cf_param = {
    "knn": 145.7,
    "shrink": 39.75,
}

item_cb_param = {
    "knn": 5.986,
    "shrink": 11.86,
}

rp3beta_param = {
    "alpha":0.7601,
    "beta":0.2004,
    "topK":225.7,
}


class Hybrid_Achille(BaseHybridRecommender):

    # MAP: 0.050377 (locale) -> MAP: 0.03630 (Pubblica) --- 5.495,0.08478,0.6595,6.516,5.236,0.09334,0.9754
    # MAP: 0.050381 (locale) -> MAP: 0.03144 (Pubblica) --- 5.401,0.1945,2.481,5.728,4.798,0.04486,2.186
    # MAP: 0.050514 (locale) -> MAP: 0.03436 (Pubblica) --- 5.916,0.1523,1.625,5.742,6.634,0.2336,1.072
    # MAP: 0.050507 (locale) -> MAP: 0.03577 (Pubblica) --- 5.495,0.1842,1.503,6.883,5.726,0.3175,1.781
    # MAP: 0.050465 (locale) -> MAP: 0.03394 (Pubblica) --- 5.817,0.09765,1.746,6.726,4.573,0.5081,1.042
    # MAP: 0.050255 (locale) -> MAP: 0.03593 (Pubblica) --- 4.403,0.0979,2.354,6.401,5.283,1.394,1.48
    # MAP: 0.050388 (locale) -> MAP: 0.03582 (Pubblica) --- 4.828,0.1012,2.271,6.89,6.926,1.131,1.301
    # MAP: 0.050444 (locale) -> MAP: 0.03593 (Pubblica) --- 4.662,0.188,1.523,7.0,6.91,1.075,1.361

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[5.46,0.9731,2.106,3.722,0.2398,2.32,1.651],
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=item_cf_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],
                   alpha_rp3beta=rp3beta_param["alpha"], beta_rp3beta=rp3beta_param["beta"], topk_rp3beta=rp3beta_param["topK"],
                   tuning=False):

        self.get_cold_users(URM)
        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all)

        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF_similarity.npz")
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning, similarity_path="/SimilarityProduct/UserCF_similarity.npz")
        self.itemContentBased.fit(URM.copy(), ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning, similarity_path="/SimilarityProduct/ItemCB_similarity.npz")
        self.elasticNet.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/Elastic_similarity.npz")
        self.RP3Beta.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/RP3Beta_similarity.npz")
        self.slim_random.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/Slim_similarity.npz")
        self.ALS.fit(URM.copy(), tuning=tuning, user_path="/SimilarityProduct/ALS_UserFactor.npz", item_path="/SimilarityProduct/ALS_ItemFactor.npz")

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.elasticNet_ratings = self.elasticNet.get_expected_ratings(user_id)
        self.RP3Beta_ratings = self.RP3Beta.get_expected_ratings(user_id)
        self.slim_ratings = self.slim_random.get_expected_ratings(user_id)
        self.ALS_ratings = self.ALS.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * self.weights[0]
        self.hybrid_ratings += self.userCF_ratings * self.weights[1]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[2]
        self.hybrid_ratings += self.elasticNet_ratings * self.weights[3]
        self.hybrid_ratings += self.RP3Beta_ratings * self.weights[4]
        self.hybrid_ratings += self.slim_ratings * self.weights[5]
        self.hybrid_ratings += self.ALS_ratings * self.weights[6]

    def extract_rating_hybrid(self):
        return self.hybrid_ratings