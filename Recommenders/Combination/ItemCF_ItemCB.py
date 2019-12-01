""" @author: Simone Lanzillotta, Stefano Martina """

from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender

class ItemCF_ItemCB(ItemKNNCFRecommender):

    def __init__(self):
        super(ItemCF_ItemCB).__init__()

    def fit(self, URM, ICM=None):
        self.URM = URM
        self.ICM_list = ICM
        self.CB = ItemCBFKNNRecommender()
        self.CB.fit(self.URM, self.ICM_list)

        similarity_object = self.compute_similarity()
        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=10, exclude_seen=True, threshold = 0.0):
        self.threshold = threshold

        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        summed_score = scores.sum(axis=0)
        if (summed_score <= self.threshold):
            return self.CB.recommend(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        ranking = scores.argsort()[::-1]
        return ranking[:at]

