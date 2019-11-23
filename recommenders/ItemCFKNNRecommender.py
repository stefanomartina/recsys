from utils.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
import os
import zipfile
import scipy.sparse as sps
from utils import evaluation
from utils import extractCSV
import Runner

class ItemCFKNNRecommender(object):

    def fit(self, URM, topK=50, shrink=100, normalize=True, similarity="tanimoto"):
        self.URM = URM
        similarity_object = Compute_Similarity_Python(self.URM, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


def get_file(file, relative_path="/../data/recommender-system-2019-challenge-polimi.zip"):
    dirname = os.path.dirname(__file__)
    dataFile = zipfile.ZipFile(dirname + relative_path)
    path = dataFile.extract(file, path=dirname + "/data")
    return open(path, 'r')


def rowSplit(rowString, token=","):
        split = rowString.split(token)
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)
        return result


def get_tuples(file):
    tuples = []
    file.seek(0)

    for line in file:
        if line != "row,col,data\n":
            tuples.append(rowSplit(line))
    return tuples


def split_dataset_loo(URM_all ):
    print('Using LeaveOneOut')
    urm = URM_all.tocsr()
    users_len = len(urm.indptr) - 1
    items_len = max(urm.indices) + 1
    urm_train = urm.copy()
    urm_test = np.zeros((users_len, items_len))
    for user_id in range(users_len):
        start_pos = urm_train.indptr[user_id]
        end_pos = urm_train.indptr[user_id + 1]
        user_profile = urm_train.indices[start_pos:end_pos]
        if user_profile.size > 0:
            item_id = np.random.choice(user_profile, 1)
            urm_train[user_id, item_id] = 0
            urm_test[user_id, item_id] = 1

    urm_test = (sps.coo_matrix(urm_test, dtype=int, shape=urm.shape)).tocsr()
    urm_train = (sps.coo_matrix(urm_train, dtype=int, shape=urm.shape)).tocsr()

    urm_test.eliminate_zeros()
    urm_train.eliminate_zeros()

    return urm_test, urm_train
