from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Hybrid.BaseHybridRecommender import BaseHybridRecommender
from Hybrid.Hybrid_Combo6 import Hybrid_Combo6
from Hybrid.Hybrid_Combo6_bis import Hybrid_Combo6_bis
import numpy as np

# This class tries to combine different kind of hybrid recommenders. Crazy change to win the competition!

class Hybrid_Combo10(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=None, weights1=None, weights2=None, tuning=False):

        self.URM = URM
        self.weights = weights
        self.weights1 = np.array(weights1)
        self.weights2 = np.array(weights2)
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.rec_for_colder.fit(self.URM, self.UCM_all)

        # Sub-Fitting
        self.Hybrid6 = Hybrid_Combo6("Combo6", UserCBFKNNRecommender())
        self.Hybrid6_bis = Hybrid_Combo6_bis("Combo6_bis", UserCBFKNNRecommender())

        self.Hybrid6.fit(self.URM, ICM_all=self.ICM_all, UCM_all=self.UCM_all, weights=self.weights1, tuning=tuning)
        self.Hybrid6_bis.fit(self.URM, ICM_all=self.ICM_all, UCM_all=self.UCM_all, weights=self.weights2, tuning=tuning)

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.Hybrid6.extract_ratings(user_id)
        self.Hybrid6.sum_ratings()

        self.Hybrid6_bis.extract_ratings(user_id)
        self.Hybrid6_bis.sum_ratings()

        self.hybrid6_ratings = self.Hybrid6.extract_rating_hybrid()
        self.hybrid6_bis_ratings = self.Hybrid6_bis.extract_rating_hybrid()

    def sum_ratings(self):
        self.hybrid_ratings = self.hybrid6_ratings * self.weights[0]
        self.hybrid_ratings += self.hybrid6_bis_ratings * self.weights[1]
