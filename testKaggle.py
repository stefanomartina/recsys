from utils import buildMatrix as bm
from utils import extractCSV as exc
import os
import numpy as np
import scipy.sparse as sps
from recommenders import RandomRecommender as rr
from utils import extractCSV as e


def rowSplit (rowString):
    split = rowString.split(",")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])

    result = tuple(split)
    return result


path_dataset = "/data/recommender-system-2019-challenge-polimi/data_train.csv"

dirname = os.path.dirname(__file__)

row = np.array(exc.open_csv(dirname + path_dataset))

saved_tuple = []

# user_id = list(row[:, 0][1:])
# item_id = list(row[:, 1][1:])

index = []
comma = [","]
appo = []

# user_id_set = set(user_id)

userList, itemList, ratings = zip(*row)
userList_unique = set(userList)

userList = list(userList[1:])
itemList = list(itemList[1:])
ratings = list(ratings[1:])

print(str(len(userList)) +" "+ str(len(itemList)) +" "+ str(len(ratings)))

# ones = list(np.ones(len(user_id)))

URM_all = sps.coo_matrix((ratings, (userList, itemList))).tocsr()

randomRecommend = rr.RandomRecommender()
randomRecommend.fit(URM_all)


for i in userList_unique:
    index = [str(i)+","]
    appo.clear()

    for i in randomRecommend.recommend(i, at=10):
        appo.append(i)

    saved_tuple.append(index + appo)

e.write_csv("results/test.csv", ["user_id", "item_list"], saved_tuple)


