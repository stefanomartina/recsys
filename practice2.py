from urllib.request import urlretrieve
import zipfile
import scipy.sparse as sps
import numpy as np

def rowSplit (rowString):
    split = rowString.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])
    split[3] = int(split[3])

    result = tuple(split)
    return result


if __name__ == '__main__':
    print("ciao")
    #/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M
    dataFile = zipfile.ZipFile("/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M/movielens_10m.zip")
    URM_path = dataFile.extract("ml-10M100K/ratings.dat", path= "/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M")

    URM_file = open(URM_path, 'r')

    print(type(URM_file))

    #for _ in range(10):
    #     print(URM_file.readline())

    URM_file.seek(0)
    numberInteractions = 0

    #for _ in URM_file:
    #    numberInteractions+=1
    #print("The number of interactions is {}: ".format(numberInteractions))

    URM_file.seek(0)
    URM_tuples = []

    for line in URM_file:
        URM_tuples.append(rowSplit(line))

    print(URM_tuples[0:10])

    userList, itemList, ratingList, timestampList = zip (*URM_tuples)

    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)
    timestampList = list(timestampList)

    print(userList[0:10])
    print(itemList[0:10])
    print(ratingList[0:10])
    print(timestampList[0:10])

    #STATISTICS PART SKIPPED!!

    URM_all = sps.coo_matrix(ratingList, (userList, itemList))
    URM_all.tocsr()

    train_test_split = 0.8
    numberInteractions = URM_all.nnz

class RandomRecommender(object):
    def fit (self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at = 5):
        recommended_items = np.random.choice(self.numItems, at)
        return recommended_items

