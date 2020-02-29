""" @author: Simone Lanzillotta, Stefano Martina """

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
import scipy.sparse as sps


RECOMMENDER_NAME = "UserCBFKNNRecommender"
SIMILARITY_PATH = "/SimilarityProduct/UserCB_similarity.npz"

class UserCBFKNNRecommender(BaseRecommender):

    def fit(self, URM, UCM_all, knn=1868, shrink=65, similarity="tversky", normalize=True, transpose=True, feature_weighting=None, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM
        self.UCM_all = UCM_all
        self.TopPopRec = TopPopRecommender()
        self.TopPopRec.fit(self.URM)

        if feature_weighting is not None:
            self.UCM_all = self.helper.feature_weight(self.UCM_all, feature_weighting)

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_stored(sps.hstack((self.UCM_all, self.URM)), RECOMMENDER_NAME, similarity_path, knn, shrink, similarity, normalize, transpose=transpose, tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(sps.hstack((self.UCM_all, self.URM)), knn, shrink, similarity, normalize,
                                                                  transpose=transpose)
        self.similarityProduct = self.W_sparse.dot(self.URM)

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_scores = self.get_expected_ratings(user_id)

        if exclude_seen:
            expected_scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]

        rec_userCBF = ranking[:at]
        if expected_scores.sum() == 0:
            return self.TopPopRec.recommend(user_id)

        return rec_userCBF