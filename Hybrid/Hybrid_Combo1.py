from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np




item_cb_param = {
    "knn": 5,
    "shrink": 100,
}

class Hybrid_Combo1(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[0.2,0.2,0.2],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all, tuning=tuning, similarity_path="/SimilarityProduct/UserCBF_similarity1.npz")


        # Sub-Fitting
        self.RP3Beta.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/RP3Beta_similarity1.npz")
        self.itemContentBased.fit(URM.copy(), ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning, similarity_path="/SimilarityProduct/ItemCB_similarity1.npz")
        self.P3Alpha.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/P3Alpha_similarity1.npz")


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.RP3Beta_ratings = self.RP3Beta.get_expected_ratings(user_id)
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.pureSVD_ratings = self.pureSVD.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.RP3Beta_ratings * self.weights[0]
        self.hybrid_ratings += self.itemContentBased_ratings * self.weights[1]
        self.hybrid_ratings += self.pureSVD_ratings * self.weights[2]

