import zipfile

import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils import extractLIST as exl
from utils import splitDataset as spl
import Runner as r
from recommenders import RandomRecommender as rr
from recommenders import TopPopRecommender as tp
from recommenders import ItemCBFKNNRecommender
from recommenders import ItemCFKNNRecommender
from recommenders import HybridItemCF_ItemCB
from utils import evaluation
import os
import scipy.sparse as sps


class Tuner():
    def __init__(self):
        self.validation_mask = None
        self.train_mask = None

        # URM ----------------
        self.userlist_urm = None
        self.itemlist_urm = None
        self.ratinglist_urm = None

        self.userlist_unique = []
        self.URM_all = None
        self.URM_train = None
        self.URM_test = None

    def get_file(self, file):
        return open("data/" + file)

    def rowSplit(self, rowString, token=","):
        split = rowString.split(token)
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)

        return result

    def get_URM_all(self):
        URM_file_name = "data_train.csv"
        URM_file = self.get_file(URM_file_name)
        URM_tuples = self.get_tuples(URM_file)

        self.get_list_URM(URM_tuples)

        self.URM_all = sps.coo_matrix((self.ratinglist_urm, (self.userlist_urm, self.itemlist_urm))).tocsr()

    def get_list_URM(self, tuples):
        userlist, itemlist, ratingslist = zip(*tuples)

        self.userlist_urm = list(userlist)
        self.itemlist_urm = list(itemlist)
        self.ratinglist_urm = list(ratingslist)

    def get_tuples(self, file, target=False):
        tuples = []
        file.seek(0)
        next(file)
        for line in file:
            if not target:
                tuples.append(self.rowSplit(line))
            if target:
                line = line.replace("\n", "")
                self.userlist_unique.append(int(line))
        return tuples

    def get_target_users(self):
        file = self.get_file("data_target_users_test.csv")
        self.get_tuples(file, target=True)

    def split_dataset_loo(self):
        print('Using LeaveOneOut')
        urm = self.URM_all.tocsr()
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
        self.URM_train = urm_train
        self.URM_test = urm_test
        print("Ended")

    def get_list_ICM(self, tuples, source):

        if source == "sub_class":
            itemlist, attributelist, presencelist = zip(*tuples)

            self.itemlist_icm = list(itemlist)
            self.attributelist_icm = list(attributelist)
            self.presencelist_icm = list(presencelist)

        if source == "asset":
            itemlist, uselesslist, assetlist = zip(*tuples)

            self.itemlist_icm_asset = list(itemlist)
            self.assetlist_icm = list(assetlist)

        if source == "price":
            itemlist, uselesslist, pricelist = zip(*tuples)

            self.itemlist_icm_price = list(itemlist)
            self.pricelist_icm = list(pricelist)

    def get_ICM(self, ICM_sub_class = True, ICM_asset = False, ICM_price = False):

        # creation of ICM using only data_ICM_sub_class
        if ICM_sub_class:
            ICM_file_name = "data_ICM_sub_class.csv"
            self.get_list_ICM(self.get_tuples(self.get_file(ICM_file_name), False), "sub_class")

            # shaping
            n_items_sub_class = self.URM_all.shape[1]
            n_sub_class = max(self.attributelist_icm) + 1
            ICM_subclass_shape = (n_items_sub_class, n_sub_class)

            self.ICM = sps.coo_matrix((self.presencelist_icm, (self.itemlist_icm, self.attributelist_icm)), shape = ICM_subclass_shape).tocsr()

        elif ICM_price:
            ICM_file_name = "data_ICM_price.csv"
            self.get_list_ICM(self.get_tuples(self.get_file(ICM_file_name), False), "price")

            # shaping and label
            le = preprocessing.LabelEncoder()
            le.fit(self.pricelist_icm)
            self.pricelist_icm = le.transform(self.pricelist_icm)
            n_items_price = self.URM_all.shape[1]
            n_price = max(self.pricelist_icm) + 1
            ICM_price_shape = (n_items_price, n_price)

            ones = np.ones(len(self.itemlist_icm_price))
            self.ICM_price = (sps.coo_matrix((ones, (self.itemlist_icm_price, self.pricelist_icm)), shape=ICM_price_shape)).tocsr()

        elif ICM_asset:
            ICM_file_name = "data_ICM_asset.csv"
            self.get_list_ICM(self.get_tuples(self.get_file(ICM_file_name), False), "asset")

            # shaping and label
            le = preprocessing.LabelEncoder()
            le.fit(self.assetlist_icm)
            self.assetlist_icm = le.transform(self.assetlist_icm)
            n_items_asset = self.URM_all.shape[1]
            n_asset = max(self.assetlist_icm) + 1
            ICM_asset_shape = (n_items_asset, n_asset)

            ones = np.ones(len(self.itemlist_icm_asset))
            self.ICM_asset = (sps.coo_matrix((ones, (self.itemlist_icm_asset, self.assetlist_icm)),
                                             shape=ICM_asset_shape)).tocsr()

    def step(self, topK, list_ICM, shrink, similarity):
        print("----------------------------------------")
        print("topk: " + str(topK) + " shrink: " + str(shrink) + " similarity: " + similarity)
        self.recommender.fit(self.URM_train, list_ICM, topK=topK, shrink=shrink, similarity=similarity, normalize=True)
        evaluation.evaluate_algorithm(self.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def run(self):
        self.recommender = HybridItemCF_ItemCB.HybridItemCF_ItemCB()
        self.get_URM_all()
        self.split_dataset_loo()
        self.get_target_users()
        self.get_ICM()
        self.get_ICM(False, True, False)
        self.get_ICM(False, False, True)
        list_ICM = [self.ICM, self.ICM_asset, self.ICM_price]

        topKs = [10, 25, 50, 75, 100]
        shrinks = [10, 25, 50, 100, 250]
        similarities = ["cosine", "pearson", "jaccard", "dice", "tanimoto"]

        for topk in topKs:
            for shrink in shrinks:
                for similarity in similarities:
                    self.step(topk, list_ICM, shrink, similarity)


if __name__ == "__main__":
    Tuner().run()
