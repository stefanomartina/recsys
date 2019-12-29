""" @author: Simone Lanzillotta, Stefano Martina """

from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np

class Hybrid_CF(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[2.991,0.09641,1.445,0.3841,2.964], tuning=False):

        self.URM = URM
        self.weights = np.array(weights)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.ICM_all, self.UCM_all, tuning=True)
        self.treshold = 1.445


        # Sub-Fitting
        self.itemCF.fit(self.URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/ItemCF_similarity.npz")
        self.userCF.fit(self.URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/UserCF_similarity.npz")
        self.ALS.fit(self.URM.copy())
        self.RP3Beta.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/RP3Beta_similarity.npz")
        self.elasticNet.fit(URM.copy(), tuning=tuning, similarity_path="/SimilarityProduct/Elastic_similarity.npz")


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemCF_ratings = self.itemCF.get_expected_ratings(user_id)
        self.userCF_ratings = self.userCF.get_expected_ratings(user_id)
        self.ALS_ratings = self.ALS.get_expected_ratings(user_id)
        self.elasticNet_ratings = self.elasticNet.get_expected_ratings(user_id)
        self.RP3Beta_ratings = self.RP3Beta.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemCF_ratings * self.weights[0]
        self.hybrid_ratings += self.userCF_ratings * self.weights[1]
        self.hybrid_ratings += self.ALS_ratings * self.weights[2]
        self.hybrid_ratings += self.elasticNet_ratings * self.weights[3]
        self.hybrid_ratings += self.RP3Beta_ratings * self.weights[4]

    def recommend(self, user_id, at=10):
        self.extract_ratings(user_id)
        self.sum_ratings()
        summed_score = self.hybrid_ratings.sum(axis=0)

        if summed_score <= self.treshold:
            return self.rec_for_colder.recommend(user_id)

        else:
            recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)
            # REMOVING SEEN
            unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                        assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

            return recommended_items[0:at]
