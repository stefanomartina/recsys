""" @author: Simone Lanzillotta, Stefano Martina """

from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender

class ItemCF_TopPop(ItemKNNCFRecommender):

    def __init__(self):
        super(ItemCF_TopPop).__init__()

    def fit(self, URM, ICM = None):
        self.URM = URM
        self.TP = TopPopRecommender()
        self.TP.fit(self.URM)

    def recommend(self, user_id, at=10, exclude_seen=True, threshold=0.3):
        self.threshold = threshold

        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()
        summed_score = scores.sum(axis=0)

        if (summed_score <= self.threshold):
            return self.TP.recommend(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]
        return ranking[:at]

