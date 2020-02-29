""" @author: Simone Lanzillotta, Stefano Martina """

from Recommenders.BaseRecommender import BaseRecommender

RECOMMENDER_NAME = "ItemKNNCFRecommender"
SIMILARITY_PATH = "/SimilarityProduct/ItemCF_similarity.npz"

class ItemKNNCFRecommender(BaseRecommender):

    def fit(self, URM, knn=10, shrink=30, similarity="jaccard", normalize=True, transpose=False, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_stored(self.URM, RECOMMENDER_NAME, similarity_path, knn, shrink,
                                                                         similarity, normalize, transpose=transpose,
                                                                         tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(self.URM, knn, shrink, similarity, normalize,
                                                                             transpose=transpose)

        self.similarityProduct = self.URM.dot(self.W_sparse)