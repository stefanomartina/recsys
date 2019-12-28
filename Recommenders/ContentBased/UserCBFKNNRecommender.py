""" @author: Simone Lanzillotta, Stefano Martina """
import scipy

from Base.BaseFunction import BaseFunction
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
import scipy.sparse as sps
import numpy as np

RECOMMENDER_NAME = "UserCBFKNNRecommender"
SIMILARITY_PATH = "/SimilarityProduct/UserCB_similarity.npz"

class UserCBFKNNRecommender():

    def __init__(self):
        self.helper = BaseFunction()

    def fit(self, URM, UCM_all, knn=2000, shrink=4.172, similarity="tversky", normalize=True, transpose=True, feature_weighting=None, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM
        self.UCM_all = UCM_all
        self.TopPopRec = TopPopRecommender()
        self.TopPopRec.fit(self.URM)
        """ 
        # Compute the extention of the UCM, adding URM and the total number of interactions of the users with the items
        user_activity = (np.asarray((self.URM).sum(axis=1)).squeeze()).astype(int)
        user_activity = list(user_activity[user_activity > 0])
        users = list(self.helper.userlist_urm)
        presence_activity = list(np.ones(len(users)))
        user_activity_adapted = users.copy()
        users_iterator = iter(users)

        head = 0
        j = 0
        i = 0
        condition = True

        while(condition):
            if(j!=len(user_activity)):
                if next(users_iterator) == head:
                    user_activity_adapted[j] = user_activity[head]
                    j += 1

                else:
                    users_iterator = iter(users)
                    head += 1
                    for i in range(0, j):
                        next(users_iterator)
            else:
                condition = False

        activity_matrix = (sps.coo_matrix((presence_activity, (users, user_activity_adapted))))
        self.UCM_all = sps.hstack((self.UCM_all, activity_matrix))
        """

        if feature_weighting is not None:
            self.UCM_all = self.helper.feature_weight(self.UCM_all, feature_weighting)

        # Compute similarity
        if tuning:
            self.W_sparse = self.helper.get_cosine_similarity_stored(sps.hstack((self.UCM_all, self.URM)), RECOMMENDER_NAME, similarity_path, knn, shrink, similarity, normalize, transpose=transpose, tuning=tuning)
        else:
            self.W_sparse = self.helper.get_cosine_similarity(sps.hstack((self.UCM_all, self.URM)), knn, shrink, similarity, normalize,
                                                                  transpose=transpose)
        self.similarityProduct = self.W_sparse.dot(self.URM)

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] -= np.inf

        return scores

    def get_expected_ratings(self, user_id):
        expected_scores = (self.similarityProduct[user_id]).toarray().ravel()
        return expected_scores

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_scores = self.get_expected_ratings(user_id)

        if exclude_seen:
            expected_scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]

        rec_userCBF = ranking[:at]
        if expected_scores.sum() == 0:
            return self.TopPopRec.recommend(user_id)

        return rec_userCBF