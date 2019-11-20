import numpy as np
from sklearn import preprocessing
from utils import extractCSV as exc
from utils import extractLIST as exl
from utils import splitDataset as spl
from recommenders import RandomRecommender as rr
from recommenders import TopPopRecommender as tp
from recommenders import createRecommendation as createRec
from recommenders import ItemCBFKNNRecommender as knn
import os
import scipy.sparse as sps


dirname = os.path.dirname(__file__)
data_path = dirname + "/data/recommender-system-2019-challenge-polimi.zip"
file_path_icm = "data_ICM_sub_class.csv"
file_path_urm = "data_train.csv"
dest_path = dirname + "/data/"


# path related to the file in which are indicated the users t be recommended
userTBR_path = dirname + "/data/data_target_users_test.csv"

# extract from dataset lists of URM
userList, itemList, ratings = exl.extractList(data_path, file_path_urm, dest_path)

# extract from dataset list of ICM
itemList_icm, classList_icm, presence_icm = exl.extractList(data_path, file_path_icm, dest_path)

# list of users to be recommended
userList_unique = exc.open_csv(userTBR_path)

# create the URM Matrix and ICM Matrix
URM_all = exl.createURM(userList, itemList, ratings)
le = preprocessing.LabelEncoder()
le.fit(classList_icm)
tagList_icm = le.transform(classList_icm)



ICM_all = exl.createICM(URM_all, itemList_icm, classList_icm, presence_icm)

# split URM in URM_train adn URM_test
URM_test, URM_train = spl.splitDataset(0.80,userList,itemList,ratings,URM_all)



""" Random Recommender """
#createRec.recommandations(rr.RandomRecommender(), userList_unique, URM_train)

""" TopPop Recommender """
#createRec.recommandations(tp.TopPopRecommender(), userList_unique, URM_train)

""" ItemCBFKNN Recommender """
recommender = knn.ItemCBFKNNRecommender(URM_train, ICM_all)
recommender.fit(shrink=0.0, topK=50)

appo = []
saved_tuple = []



# create the result.csv
for i in userList_unique:
    index = [i[0] + ","]
    appo.clear()
    for i in recommender.recommend(int(i[0]), 10):
        appo.append(i)

    saved_tuple.append(index + appo)

exc.write_csv(saved_tuple, "ItemCBFKNN")



