from utils import buildMatrix as bm
from utils import extractCSV as exc
import os
import numpy as np
import scipy.sparse as sps
from recommenders import RandomRecommender as rr
from utils import extractCSV as e


def listToString(s):
    str1 = ""
    for ele in s:
        str1 += " " + str(ele)
    return str1

dirname = os.path.dirname(__file__)
row = np.array(exc.open_csv(dirname + "/data/OLD_DATASET/train.csv"))

saved_tuple = []
to_be_printed = []

playlist = row[:, 0][1:]
song = row[:, 1][1:]

index = []
comma = [","]
appo = []

playlist_set = set(playlist)
list_of_ones = np.ones(np.shape(playlist)[0])
print(np.shape(playlist)[0])

URM_all = sps.coo_matrix((list_of_ones, (playlist, song)), shape=(np.shape(playlist)[0], np.shape(song)[0]))

randomRecommend = rr.RandomRecommender()
randomRecommend.fit(URM_all)


for i in playlist_set:
    index = [str(i)+","]
    appo.clear()
    for i in randomRecommend.recommend(i, at=10):
        appo.append(i)

    saved_tuple.append(index + appo)

e.write_csv("test.csv", ["playlist_id", "track_ids"], saved_tuple)


