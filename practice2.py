from urllib.request import urlretrieve
import zipfile
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as pyplot

class RandomRecommender(object):
    def fit(self, URM_train):
        self.numItems = URM_train.shape[0]

    def recommend(self, user_id, at = 5):
        recommended_items = np.random.choice(self.numItems, at)
        return recommended_items

def rowSplit (rowString):
    split = rowString.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])
    split[3] = int(split[3])

    result = tuple(split)
    return result

def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score

def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score

def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score

def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0

    num_eval = 0

    for user_id in userList_unique:

        relevant_items = URM_test[user_id].indices

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_MAP += MAP(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))


class TopPopRecommender(object):
    def fit(self, URM_train):
        self.URM_train = URM_train
        itemPopularity = (URM_train>0).sum(axis=0)
        itemPopularity = np.array(itemPopularity).squeeze()
        self.popularItems = np.argsort(itemPopularity)
        self.popularItems = np.flip(self.popularItems, axis = 0)

    def recommend(self, user_id, at=5, remove_seen=True):
        if remove_seen:
            unseen_items_mask = np.in1d(self.popularItems, self.URM_train[user_id].indices,assume_unique=True, invert = True)
            unseen_items = self.popularItems[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popularItems[0:at]
        return recommended_items


if __name__ == '__main__':
    # /Users/Stefano/PycharmProjects/recsys/data/Movielens_10M
    dataFile = zipfile.ZipFile("/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M/movielens_10m.zip")
    URM_path = dataFile.extract("ml-10M100K/ratings.dat", path= "/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M")

    URM_file = open(URM_path, 'r')
    # print(type(URM_file))

    # SEEK -> Sets the file’s current position
    URM_file.seek(0)
    numberInteractions = 0

    for _ in URM_file:
        numberInteractions+=1
    # print("The number of interactions is {}: ".format(numberInteractions))

    URM_file.seek(0)
    URM_tuples = []

    for line in URM_file:
        URM_tuples.append(rowSplit(line))

    print(URM_tuples[0:10])

    # The zip() function returns a zip object, which is an iterator of tuples where the first item
    # in each passed iterator is paired together,
    #   and then the second item in each passed iterator are paired together etc.
    userList, itemList, ratingList, timestampList = zip (*URM_tuples)

    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)
    timestampList = list(timestampList)

    # print(userList[0:10])
    # print(itemList[0:10])
    # print(ratingList[0:10])
    # print(timestampList[0:10])

    userList_unique = list(set(userList))
    itemList_unique = list(set(itemList))
    numUsers = len(userList_unique)
    numItems = len(itemList_unique)

    # print("Number of items\t {}, Number of users\t {}".format(numItems, numUsers))
    # print("Max ID items\t {}, Max Id users\t {}\n".format(max(itemList_unique), max(userList_unique)))
    # print("Average interactions per user {:.2f}".format(numberInteractions / numUsers))
    # print("Average interactions per item {:.2f}\n".format(numberInteractions / numItems))
    # print("Sparsity {:.2f} %".format((1 - float(numberInteractions) / (numItems * numUsers)) * 100))


    timestamp_sorted = list(timestampList)
    timestamp_sorted.sort();

    # pyplot.plot(timestamp_sorted, 'ro')
    # pyplot.ylabel('Timestamp')
    # pyplot.xlabel('Item Index')
    # pyplot.show()

    URM_all = sps.coo_matrix((ratingList, (userList, itemList)))

    URM_all.tocsr()

    # (URM_all > 0 ) returns a boolean matrix with the same dimensions of the URM_all
    # by summing all the elements, assuming that true is one, we obtain the popularity of
    itemPopularity = (URM_all>0).sum(axis=0)

    itemPopularity = np.array(itemPopularity).squeeze()
    itemPopularity = np.sort(itemPopularity)

    # pyplot.plot(itemPopularity, 'ro')
    # pyplot.ylabel('Num interactions')
    # pyplot.xlabel('Item Index')
    # pyplot.show()

    tenPercent = int(numItems / 10)

    # print("Average per-item interactions over the whole dataset {:.2f}".format(itemPopularity.mean()))
    # print("Average per-item interactions for the top 10% popular items {:.2f}".format(itemPopularity[-tenPercent].mean()))
    # print("Average per-item interactions for the least 10% popular items {:.2f}".format(itemPopularity[:tenPercent].mean()))
    # print("Average per-item interactions for the median 10% popular items {:.2f}".format(itemPopularity[int(numItems * 0.45):int(numItems * 0.55)].mean()))
    # print("Number of items with zero interactions {}".format(np.sum(itemPopularity == 0)))

    itemPopularityNonzero = itemPopularity[itemPopularity > 0]
    tenPercent = int(len(itemPopularityNonzero) / 10)

    print("Average per-item interactions over the whole dataset {:.2f}".format(itemPopularityNonzero.mean()))
    print("Average per-item interactions for the top 10% popular items {:.2f}".format(itemPopularityNonzero[-tenPercent].mean()))
    print("Average per-item interactions for the least 10% popular items {:.2f}".format(itemPopularityNonzero[:tenPercent].mean()))
    print("Average per-item interactions for the median 10% popular items {:.2f}".format(itemPopularityNonzero[int(numItems * 0.45):int(numItems * 0.55)].mean()))

    # pyplot.plot(itemPopularityNonzero, 'ro')
    # pyplot.ylabel('Num Interactions ')
    # pyplot.xlabel('Item Index')
    # pyplot.show()

    train_test_split = 0.80

    # NNZ -> Counts the number of non-zero values in the array a.
    numInteractions = URM_all.nnz

    # Train mask -> a mask to mask the cells of the train set
    # array([ True,  True,  True])
    # traing_mask = np.random.choice([True, False],3, p=[t, 1-t])
    # print traing_mask
    #   array([ True, False,  True])
    # print h[traing_mask]    <-- masked output
    #   array(['2', '4'], dtype='<U1')
    train_mask = np.random.choice([True, False], numInteractions, p=[train_test_split, 1-train_test_split])

    userList = np.array(userList)
    itemList = np.array(itemList)
    ratingList = np.array(ratingList)

    URM_train = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
    URM_train = URM_train.tocsr()

    test_mask = np.logical_not(train_mask)
    URM_test = sps.coo_matrix((ratingList[train_mask], (userList[train_mask], itemList[train_mask])))
    URM_test = URM_test.tocsr()


    user_id = userList_unique[1]

    # Random recommender
    # randomRecommend = RandomRecommender()
    # randomRecommend.fit(URM_train)

    # recommended_item = randomRecommend.recommend(user_id, at=5)
    # print(recommended_item)

    # relevant_items = URM_test[user_id].indices
    # print(relevant_items)

    # is_relevant = np.in1d(recommended_item, relevant_items, assume_unique=True)
    # print(is_relevant)

    # evaluate_algorithm(URM_test, randomRecommend)
    # End of Random recommender

    # TopPop Recommender
    # topPopRecommender_removeSeen = TopPopRecommender()
    # topPopRecommender_removeSeen.fit(URM_train)

    # for user_id in userList_unique[0:10]:
    #    print(topPopRecommender_removeSeen.recommend(user_id, at=5))

    # evaluate_algorithm(URM_test, topPopRecommender_removeSeen)

    # End of TopPop Recommender

    # Global effects recommender
    globalAverage = np.mean(URM_train.data)
    print(globalAverage)

    # Unbias the data
    URM_train_unbiased = URM_train.copy()
    URM_train_unbiased.data -= globalAverage

    item_mean_rating = URM_train_unbiased.mean(axis=0)

    item_mean_rating = np.array(item_mean_rating).squeeze()
    item_mean_rating = np.sort(item_mean_rating[item_mean_rating!=0])

    pyplot.plot(item_mean_rating, 'ro')
    pyplot.ylabel("Item bias")
    pyplot.xlabel("Item Index")
    pyplot.show()


