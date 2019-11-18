import numpy as np
import scipy.sparse as sps

""" 
    Method to extract Train Set and Test Set from formatted Dataset 
        - percentage: size of data related to Train Set
        - URM_tuple: list of tuples extracted from the dataset (e.g. userList, itemList, ratingList)     
"""

def splitDataset(percentage, user_list, item_list, rating_list, URM_all):

    user_list = np.array(user_list)
    item_list = np.array(item_list)
    rating_list = np.array(rating_list)

    # Count the number of relevant interaction between user and item
    numInteractions = URM_all.nnz

    # Create a random mask to put on URM
    train_mask = np.random.choice([True, False], numInteractions, p=[percentage, 1 - percentage])

    # Create URM_train with all list of original dataset randomly choosed
    URM_train = sps.coo_matrix((rating_list[train_mask], (user_list[train_mask], item_list[train_mask]))).tocsr()

    # Create URM_test using the logical_not operator applied on the train_mask (to keep the remain
    # part of dataset as test set)
    test_mask = np.logical_not(train_mask)
    URM_test = sps.coo_matrix((rating_list[test_mask], (user_list[test_mask], item_list[test_mask]))).tocsr()

    return URM_test, URM_train



