from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np

class Hybrid_CB(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None, weights=None, tuning=False):

        self.URM = URM

        self.ICM_all = ICM_all
        self.UCM_all = UCM_all

        # Sub-Fitting
        self.itemContentBased.fit(self.URM.copy(), self.ICM_all, tuning=tuning, similarity_path="/SimilarityProduct/ItemCBF_similarity_HCB.npz")
        self.userContentBased.fit(self.URM.copy(), self.UCM_all, tuning=tuning, similarity_path="/SimilarityProduct/UserCBF_similarity_HCB.npz")


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def extract_ratings(self, user_id):
        self.itemContentBased_ratings = self.itemContentBased.get_expected_ratings(user_id)
        self.userContentBased_ratings = self.userContentBased.get_expected_ratings(user_id)

    def sum_ratings(self):
        self.hybrid_ratings = self.itemContentBased_ratings * 4.793
        self.hybrid_ratings += self.userContentBased_ratings * 0.3428

    def recommend(self, user_id, at=10):
        self.extract_ratings(user_id)
        self.sum_ratings()

        recommended_items = np.flip(np.argsort(self.hybrid_ratings), 0)
        # REMOVING SEEN
        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                        assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]

        return recommended_items[0:at]
