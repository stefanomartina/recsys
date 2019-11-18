from utils import extractCSV as exc
from utils import extractLIST as exl
from utils import splitDataset as spl
from recommenders import RandomRecommender as rr
import os
import numpy as np
import scipy.sparse as sps

dirname = os.path.dirname(__file__)
data_path = dirname + "/data/recommender-system-2019-challenge-polimi.zip"
file_path = "data_train.csv"
dest_path = dirname + "/data/"

# path related to the file in which are indicated the users t be recommended
userTBR_path = dirname + "/data/data_target_users_test.csv"

# extract from dataset lists
userList, itemList, ratings = exl.extractList(data_path, file_path, dest_path)

# list of users to be recommended
userList_unique = exc.open_csv(userTBR_path)

saved_tuple = []

# support to create the the result.csv file
index = []
comma = [","]
appo = []

# create the URM Matrix
URM_all = sps.coo_matrix((ratings, (userList, itemList))).tocsr()


""" Random Recommender """

randomRecommend = rr.RandomRecommender()
randomRecommend.fit(URM_all)

# create the result.csv
for i in userList_unique:
    index = [str(i) + ","]
    appo.clear()

    for i in randomRecommend.recommend(i, at=10):
        appo.append(i)

    saved_tuple.append(index + appo)

exc.write_csv("test.csv", ["user_id", "item_list"], saved_tuple)