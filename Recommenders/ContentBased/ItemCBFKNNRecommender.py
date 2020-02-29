""" @author: Simone Lanzillotta, Stefano Martina """

import scipy.sparse as sps
import numpy as np
from Base.BaseFunction import BaseFunction
from Recommenders.BaseRecommender import BaseRecommender

RECOMMENDER_NAME = "ItemCBFKNNRecommender"
SIMILARITY_PATH = "/SimilarityProduct/ItemCB_similarity.npz"

class ItemCBFKNNRecommender(BaseRecommender):

    def fit(self, URM, ICM_all, knn=5, shrink=100, similarity="cosine", normalize=True, transpose=True, feature_weighting = None, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM
        self.ICM_all = ICM_all

        self.ICM_all = sps.hstack((self.ICM_all.tocoo(), self.URM.T.tocoo()))
        self.ICM_all = self.ICM_all.tocsr()

        if feature_weighting is not None:
            self.ICM_all = self.helper.feature_weight(self.ICM_all, feature_weighting)

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_stored(self.ICM_all, RECOMMENDER_NAME, similarity_path, knn, shrink, similarity, normalize, transpose=transpose, tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(self.ICM_all, knn, shrink, similarity, normalize, transpose=transpose)

        self.similarityProduct = self.URM.dot(self.W_sparse)
