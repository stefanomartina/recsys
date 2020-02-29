""" @author: Simone Lanzillotta, Stefano Martina """

from Recommenders.BaseRecommender import BaseRecommender

RECOMMENDER_NAME = "UserKNNCFRecommender"
SIMILARITY_PATH = "/SimilarityProduct/UserCF_similarity.npz"

class UserKNNCFRecommender(BaseRecommender):

    def fit(self, URM, knn=600, shrink=0, similarity="cosine", normalize=True, transpose=True, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_stored(self.URM, RECOMMENDER_NAME, similarity_path, knn, shrink,
                                                                         similarity, normalize, transpose=transpose,
                                                                         tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(self.URM, knn, shrink, similarity, normalize,
                                                                  transpose=transpose)

        self.similarityProduct = self.W_sparse.dot(self.URM)
