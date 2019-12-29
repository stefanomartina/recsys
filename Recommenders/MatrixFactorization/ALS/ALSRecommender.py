import os

import numpy as np
import implicit

from Base.BaseFunction import BaseFunction

RECOMMENDER_NAME = "ALSRecommender"
USER_PATH = "/SimilarityProduct/ALS_UserFactor.npz"
ITEM_PATH = "/SimilarityProduct/ALS_ItemFactor.npz"

class AlternatingLeastSquare:

    def __init__(self, n_factors=400, regularization=0.1104, iterations=50):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.helper = BaseFunction()

    def run_fit(self):
        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=self.n_factors, regularization=self.regularization,
                                                     iterations=self.iterations)

        alpha_val = 24
        # Calculate the confidence by multiplying it by our alpha value.
        data_conf = (self.sparse_item_user * alpha_val).astype('double')

        # Fit the model
        model.fit(data_conf)

        # Get the user and item vectors from our trained model
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def fit(self, URM, tuning=False, user_path=USER_PATH, item_path=ITEM_PATH):
        self.URM = URM
        self.sparse_item_user = self.URM.T

        if tuning:
            if not os.path.exists(os.getcwd() + user_path) and not os.path.exists(os.getcwd() + user_path):
                self.run_fit()
                self.helper.export_nparr(user_path, self.user_factors)
                self.helper.export_nparr(item_path, self.item_factors)
            self.user_factors = self.helper.import_nparr(user_path)
            self.item_factors = self.helper.import_nparr(item_path)
        else:
            self.run_fit()

    def get_expected_ratings(self, user_id):
        scores = np.dot(self.user_factors[user_id], self.item_factors.T)
        return np.squeeze(scores)

    def recommend(self, user_id, at=10):

        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices,
                                    assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]