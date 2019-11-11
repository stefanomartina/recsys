class RandomRecommender(object):
    def fit(self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at = 5):
        recommended_items = np.random.choice(self.numItems, at)
        return recommended_items