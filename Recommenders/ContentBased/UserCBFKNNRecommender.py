""" @author: Simone Lanzillotta, Stefano Martina """
import scipy

from Base.BaseFunction import BaseFunction
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

        return ranking[:at]
        '''return self.merge_rec(self.TopPop.recommend(user_id), ranking[:at])'''

        #return ranking[:at]

    #METHODS added trying to implement smart merge of recommendation lists
    '''def my_index(self, l, item):
        for i in range(len(l)):
            if (item == l[i]).all():
                return i
        return np.inf

    def getList(self, dict):
        list = []
        for key in dict.keys():
            list.append(key)

        return list

    def merge_rec(self, TopPop_rec, UserCBF_rec,):
        dict = {}
        elem_set = list(set(TopPop_rec) | set(UserCBF_rec))
        medium_dict = {}

        for elem in elem_set:
            index_top_pop = self.my_index(TopPop_rec, elem)
            index_user_cbf = self.my_index(UserCBF_rec, elem)
            dict.update({elem: [index_top_pop, index_user_cbf]})

        for elem in dict.keys():
            medium_dict.update({elem: np.median(dict[elem])})

        sorted_medium_dict = {k: v for k, v in sorted(medium_dict.items(), key=lambda item: item[1])}

        return self.getList(sorted_medium_dict)[:10]'''