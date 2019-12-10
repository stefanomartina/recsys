from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import UserCBFKNNRecommender, ItemCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
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

        # TopPopRecommender (for the cold user)
        self.rec_for_colder = rec_for_colder

        # User Content Based Recommender
        self.userContentBased = UserCBFKNNRecommender.UserCBFKNNRecommender()

        # Item Content Based Recommender
        self.itemContentBased = ItemCBFKNNRecommender.ItemCBFKNNRecommender()

        # User Collaborative Filtering Recommender
        self.userCF = UserKNNCFRecommender()

        # Item Collaborative Filtering Recommender
        self.itemCF = ItemKNNCFRecommender()

        # Slim Recommender
        self.slim_random = SLIM_BPR_Cython(epochs=slim_param["epochs"], topK=slim_param["topK"])

        # Ratings from each available algorithm
        self.userContentBased_ratings = None
        self.itemContentBased_ratings = None
        self.itemCF_ratings = None
        self.userCF_ratings = None
        self.icf_tp_combo_ratings = None
        self.ucf_tp_combo_ratings = None
        self.cf_cb_combo_ratings = None
        self.slim_ratings = None


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

        else:

            recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)
            # REMOVING SEEN
            unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                        assume_unique=True, invert=True)
            recommended_items = recommended_items[unseen_items_mask]

            return recommended_items[0:at]






