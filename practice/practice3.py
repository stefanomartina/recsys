from utils import buildMatrix as bm, stats as st
import numpy as np
import scipy.sparse as sps
from sklearn import preprocessing

from sys import path
path.insert(0, '/Users/Stefano/recsys_repo/RecSys_Course_2018')
try:
    from Notebooks_utils.data_splitter import train_test_holdout
    from Notebooks_utils.evaluation_function import evaluate_algorithm
    from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
except ImportError as e:
    print(e)
    print('You may need to update your path')
    print(path)
    exit(1)

class ItemCBFKNNRecommender(object):
    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.ICM.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toArray().revel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id+1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] -= np.inf

        return scores

if __name__ == '__main__':
    path = "/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M/movielens_10m.zip"
    wte = "ml-10M100K/ratings.dat"
    dest_path = "/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M"

    #Building the URM matrix
    userList_urm, itemList_urm, ratingList_urm, timestampList_urm  = bm.build(path, wte, dest_path, True)

    URM_all = sps.coo_matrix((ratingList_urm, (userList_urm, itemList_urm)))

    st.list_ID_stats(userList_urm, "User")
    st.list_ID_stats(itemList_urm, "Item")

    wte = "ml-10M100K/tags.dat"

    userList_icm, itemList_icm, tagList_icm, timestampList_icm = bm.build(path, wte, dest_path, False)

    le = preprocessing.LabelEncoder()
    le.fit(tagList_icm)

    n_items = URM_all.shape[1]
    n_tags = max(tagList_icm) + 1
    ICM_shape = (n_items, n_tags)

    ones = np.ones(len(tagList_icm))

    ICM_all = sps.coo_matrix((ones, (itemList_icm, tagList_icm)), shape=ICM_shape)
    ICM_all = ICM_all.tocsr()


    st.list_ID_stats(userList_icm, "User_icm")
    st.list_ID_stats(itemList_urm, "Item_icm")

    numTags = len(set(tagList_icm))

    tagList_icm = le.transform(tagList_icm)

    URM_train, URM_test = train_test_holdout(URM_all, train_perc=0.8)

    recommender = ItemCBFKNNRecommender(URM_train, ICM_all)
    recommender.fit()







