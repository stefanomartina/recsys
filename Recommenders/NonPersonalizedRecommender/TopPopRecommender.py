import numpy as np

class TopPopRecommender():
    RECOMMENDER_NAME = "TopPopRecommender"

    def fit(self, URM_train, tuning=False):
        print("Fitting Top Pop Recommender...")

        self.URM_train = URM_train
        itemPopularity = (URM_train>0).sum(axis=0)
        self.itemPopularity = np.array(itemPopularity).squeeze()
        self.popularItems = np.argsort(self.itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis=0)

    def recommend(self, user_id, at=10, remove_seen=True):
        if remove_seen:
            unseen_items_mask = np.in1d(self.popularItems, self.URM_train[user_id].indices,assume_unique=True, invert = True)
            unseen_items = self.popularItems[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popularItems[0:at]
        return recommended_items
