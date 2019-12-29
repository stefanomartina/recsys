""" @author: Simone Lanzillotta, Stefano Martina """

from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import UserCBFKNNRecommender, ItemCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.Slim.SlimElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.MatrixFactorization.PureSVD.PureSVDRecommender import PureSVDRecommender
from Recommenders.GraphBased.P3AlphaRecommender import P3AlphaRecommender
from Recommenders.GraphBased.RP3BetaRecommender import RP3BetaRecommender
from Recommenders.MatrixFactorization.ALS.ALSRecommender import AlternatingLeastSquare
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
import numpy as np

slim_param = {
    "epochs": 200,
    "topK": 10,
}

class BaseHybridRecommender(object):

    #######################################################################################
    #                                  INIT ALGORITHM                                     #
    #######################################################################################

    def __init__(self, combination, rec_for_colder):

        self.hybrid_ratings = None
        self.URM = None
        self.list_UCM = None
        self.list_ICM = None
        self.combination = combination
        self.merge_index = 2
        self.treshold = None

        # Recommender for the cold user
        self.rec_for_colder = rec_for_colder

        # User Content Based Recommender
        self.userContentBased = UserCBFKNNRecommender()

        # Item Content Based Recommender
        self.itemContentBased = ItemCBFKNNRecommender.ItemCBFKNNRecommender()

        # User Collaborative Filtering Recommender
        self.userCF = UserKNNCFRecommender()

        # Item Collaborative Filtering Recommender
        self.itemCF = ItemKNNCFRecommender()

        # Slim Recommender
        self.slim_random = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"])

        # ElasticNet
        self.elasticNet = SLIMElasticNetRecommender()

        # PureSVD Recommender
        self.pureSVD = PureSVDRecommender()

        # P3Alpha Recommender
        self.P3Alpha = P3AlphaRecommender()

        # RP3Beta Recommender
        self.RP3Beta = RP3BetaRecommender()

        # ALS Recommender
        self.ALS = AlternatingLeastSquare()

        # Ratings from each available algorithm
        self.userContentBased_ratings = None
        self.itemContentBased_ratings = None
        self.itemCF_ratings = None
        self.userCF_ratings = None
        self.cf_cb_combo_ratings = None
        self.slim_ratings = None
        self.elasticNet_ratings = None
        self.pureSVD_ratings = None
        self.P3Alpha_ratings = None
        self.RP3Beta_ratings = None
        self.ALS_ratings = None

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, weights, ICM_all, UCM_all):
        pass

    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        pass

    def sum_ratings(self):
        pass

    #######################################################################################
    #                                    RECOMMENDING                                     #
    #######################################################################################

    def recommend(self, user_id, at=10):
        self.extract_ratings(user_id)
        self.sum_ratings()
        summed_score = self.hybrid_ratings.sum(axis=0)

        if summed_score == 0:
            return self.rec_for_colder.recommend(user_id)

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)
        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                        assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]

    def get_cold_users(self, URM):
        cold_users_mask = np.ediff1d(URM.tocsr().indptr) == 0
        cold_users = np.arange(URM.shape[0])[cold_users_mask]

        self.cold_users = cold_users
