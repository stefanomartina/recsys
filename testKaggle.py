from utils import extractCSV as exc
from utils import extractLIST as exl
from utils import splitDataset as spl
from recommenders import RandomRecommender as rr
from recommenders import TopPopRecommender as tp
from recommenders import createRecommendation as createRec
import os
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

# create the URM Matrix
URM_all = sps.coo_matrix((ratings, (userList, itemList))).tocsr()

# split URM in URM_train adn URM_test
URM_test, URM_train = spl.splitDataset(0.80,userList,itemList,ratings,URM_all)


""" Random Recommender """
createRec.recommandations(rr.RandomRecommender(), userList_unique, URM_train)

""" TopPop Recommender """
createRec.recommandations(tp.TopPopRecommender(), userList_unique, URM_train)

