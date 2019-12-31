""" @author: Simone Lanzillotta, Stefano Martina """

from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np

class Hybrid_Achille_Tuning(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=[5.916,0.1523,1.625,5.742,6.634,0.2336,1.072],
            ItemCF=None, UserCF=None, ItemCB=None, ElasticNet=None, RP3=None, Slim=None, ALS=None,
            tuning=False):

        self.get_cold_users(URM)
        self.URM = URM
        self.weights = np.array(weights)

        # Recommender
        self.itemCF = ItemCF
        self.userCF = UserCF
        self.itemContentBased = ItemCB
        self.elasticNet = ElasticNet
        self.RP3Beta = RP3
        self.slim_random = Slim
        self.ALS = ALS

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