""" @author: Simone Lanzillotta, Stefano Martina """

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

slim_param = {
    "epochs": 200,
    "topK": 10,
}


class Hybrid_Achille(BaseHybridRecommender):

    # MAP: 0.050377 (locale) -> MAP: 0.03630 (Pubblica) --- 5.495,0.08478,0.6595,6.516,5.236,0.09334,0.9754
    # MAP: 0.050381 (locale) -> MAP:         (Pubblica) --- 5.401,0.1945,2.481,5.728,4.798,0.04486,2.186
    # MAP: 0.050514 (locale) -> MAP: 0.03436 (Pubblica) --- 5.916,0.1523,1.625,5.742,6.634,0.2336,1.072
    # MAP: 0.050465 (locale) -> MAP: 0.03394 (Pubblica) --- 5.817,0.09765,1.746,6.726,4.573,0.5081,1.042

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[5.401,0.1945,2.481,5.728,4.798,0.04486,2.186],
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