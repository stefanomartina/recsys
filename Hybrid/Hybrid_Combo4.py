from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np


item_cf_param = {
    "knn": 10,
    "shrink": 30,
}

#weights=[0.3362, 0.8046]
class Hybrid_Combo4(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, knn_itemcf=item_cf_param["knn"],
                   shrink_itemcf=item_cf_param["shrink"], weights=[0.1, 0.9],
                   tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all)

        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning, similarity_path="/SimilarityProduct/ItemCF_similarity4.npz")
        self.RP3Beta.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/RP3Beta_similarity4.npz")
        self.pureSVD.fit(URM.copy())


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.RP3Beta_ratings = self.RP3Beta.get_expected_ratings(user_id)
        self.pureSVD_ratings = self.pureSVD.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * (self.weights[0])
        self.hybrid_ratings += self.RP3Beta_ratings * (self.weights[1])
        self.hybrid_ratings += self.pureSVD_ratings * (self.weights[2])

