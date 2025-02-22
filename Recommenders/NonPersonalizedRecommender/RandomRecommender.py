import numpy as np

class RandomRecommender():
    RECOMMENDER_NAME = "RandomRecommender"

    def fit(self, URM_train):
        print("Fitting Random Recommender...")

        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at = 10):
        recommended_items = np.random.choice(self.numItems, at)
        return recommended_items
