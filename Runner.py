from sklearn import preprocessing
from tqdm import tqdm
from recommenders import RandomRecommender
from recommenders import TopPopRecommender
from recommenders import ItemCBFKNNRecommender
from recommenders import ItemCFKNNRecommender
from recommenders.SlimRecommender.Cython import SLIM_BPR_Cython
from recommenders import HybridItemCF_ItemCB
from recommenders import UserCBFKNNRecommender
from recommenders.SLIM_BPR import SLIM_BPR
# from recommenders.MatrixFactorizationRecommenders import PureSVDRecommender
# from recommenders.MatrixFactorizationRecommenders.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython
from utils import evaluation
from Base import Incremental_Training_Early_Stopping
import scipy.sparse as sps
import numpy as np
import time
import csv
import argparse

class Runner:

    def __init__(self, recommender, name, evaluate=True):
        print("Evaluate: " + str(evaluate))
        self.recommender = recommender
        self.evaluate = evaluate
        self.userlist_unique = []
        self.name = name

        self.validation_mask = None
        self.train_mask = None

        # URM ----------------
        self.userlist_urm = None
        self.itemlist_urm = None
        self.ratinglist_urm = None

        self.URM_all = None
        self.URM_train = None
        self.URM_test = None

        # ICM ----------------
        self.ICM = None
        self.itemlist_icm = None
        self.attributelist_icm = None
        self.presencelist_icm = None

        # ICM_asset ----------
        self.ICM_asset = None
        self.itemlist_icm_asset = None
        self.assetlist_icm = None

        # ICM_price ----------
        self.ICM_price = None
        self.itemlist_icm_price = None
        self.pricelist_icm = None

        # UCM_age ------------
        self.UCM_age = None
        self.userlist_ucm_age = None
        self.agelist_ucm = None
        self.presencelist_ucm_age = None

        # UCM_region ---------
        self.UCM_region = None
        self.userlist_ucm_region = None
        self.regionlist_ucm = None
        self.presencelist_ucm_region = None

    def rowSplit(self, rowString, token=","):
        split = rowString.split(token)
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)

        return result

    def write_csv(self, rows, name, fields=["user_id", "item_list"]):
        timestr = time.strftime("%Y-%m-%d_%H.%M.%S")
        file_path = "results/" + name + "-" + timestr + ".csv"

        with open(file_path, 'w') as csv_file:
            csv_write_head = csv.writer(csv_file, delimiter=',')
            csv_write_head.writerow(fields)
            csv_write_content = csv.writer(csv_file, delimiter=' ')
            csv_write_content.writerows(rows)

    def get_file(self, file):
        return open("data/" + file)

    def get_target_users(self):
        file = self.get_file("data_target_users_test.csv")
        self.get_tuples(file, target=True)

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

    def get_list_URM(self, tuples):
        userlist, itemlist, ratingslist = zip(*tuples)

        self.userlist_urm = list(userlist)
        self.itemlist_urm = list(itemlist)
        self.ratinglist_urm = list(ratingslist)

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

    def get_list_UCM(self, tuples, source):
        if source == "age":
            itemlist, attributelist, presencelist = zip(*tuples)

            self.userlist_ucm_age = list(itemlist)
            self.agelist_ucm = list(attributelist)
            self.presencelist_ucm_age = list(presencelist)

        if source == "region":
            itemlist, attributelist, presencelist = zip(*tuples)

            self.userlist_ucm_region = list(itemlist)
            self.regionlist_ucm = list(attributelist)
            self.presencelist_ucm_region = list(presencelist)

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


    def get_URM(self):
        URM_file_name = "data_train.csv"
        URM_file = self.get_file(URM_file_name)
        URM_tuples = self.get_tuples(URM_file)

        self.get_list_URM(URM_tuples)

        self.URM_all = sps.coo_matrix((self.ratinglist_urm, (self.userlist_urm, self.itemlist_urm))).tocsr()

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

    def get_UCM(self, UCM_age = True, UCM_region = False):
        if UCM_region:
            UCM_file_name = "data_UCM_region.csv"
            self.get_list_UCM(self.get_tuples(self.get_file(UCM_file_name), False), "region")
            self.UCM_region = sps.coo_matrix(
                (self.presencelist_ucm_region, (self.userlist_ucm_region, self.regionlist_ucm))).tocsr()

        if UCM_age:
            UCM_file_name = "data_UCM_age.csv"
            self.get_list_UCM(self.get_tuples(self.get_file(UCM_file_name), False), "age")
            self.UCM_age = sps.coo_matrix((self.presencelist_ucm_age, (self.userlist_ucm_age, self.agelist_ucm))).tocsr()



    def fit_recommender(self, requires_icm = False, requires_ucm = False):
        print("Fitting model...")
        if not self.evaluate:
            if requires_icm:
                list_ICM = [self.ICM, self.ICM_asset, self.ICM_price]
                self.recommender.fit(self.URM_all, list_ICM)
            elif requires_ucm:
                list_UCM = [self.UCM_age, self.UCM_region]
                self.recommender.fit(self.URM_all, list_UCM)
            else:
                self.recommender.fit(self.URM_all)
        else:
            self.split_dataset_loo()
            if requires_icm:
                list_ICM = [self.ICM, self.ICM_asset, self.ICM_price]
                self.recommender.fit(self.URM_train, list_ICM)
            elif requires_ucm:
                list_UCM = [self.UCM_age, self.UCM_region]
                self.recommender.fit(self.URM_train, list_UCM)
            else:
                self.recommender.fit(self.URM_train)
        print("Model fitted")

    def run_recommendations(self):
        recommendations = []
        saved_tuple = []
        print("Computing recommendations...")
        for user in tqdm(self.userlist_unique):
            index = [str(user) + ","]
            recommendations.clear()

            for recommendation in self.recommender.recommend(user):
                recommendations.append(recommendation)
            saved_tuple.append(index + recommendations)
        print("Recommendations computed")
        print("Printing csv...")
        self.write_csv(saved_tuple, self.name)
        print("Ended")
        return saved_tuple

    def run(self, requires_icm=False, requires_ucm=False):
        self.get_URM()
        if requires_icm:
            self.get_ICM()
            self.get_ICM(False, True, False)
            self.get_ICM(False, False, True)
        elif requires_ucm:
            self.get_UCM(False, True)
            self.get_UCM()

        self.get_target_users()
        self.fit_recommender(requires_icm, requires_ucm)
        self.run_recommendations()
        if self.evaluate:
            evaluation.evaluate_algorithm(self.URM_test, self.recommender, at=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'top-pop', 'ItemCBF', 'HybridItemCBF', 'HybridItemCF_ItemCB',
                                                'UserCBFKNN', 'ItemCF', 'SlimBPRCython_Hybrid', 'PureSVD', 'MF_BPR_Cython', 'SlimBPR'])
    parser.add_argument('--eval', action="store_true")
    args = parser.parse_args()
    requires_icm = False
    requires_ucm = False
    recommender = None

    if args.recommender == 'random':
        print("random selected")
        recommender = RandomRecommender.RandomRecommender()

    if args.recommender == 'top-pop':
        print("top-pop selected")
        recommender = TopPopRecommender.TopPopRecommender()

    if args.recommender == 'ItemCBF':
        print("ItemCBF selected")
        recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
        requires_icm = True

    if args.recommender == "HybridItemCBF":
        print("HybridItemCBF selected")
        recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
        requires_icm = True

    if args.recommender == "HybridItemCF_ItemCB":
        print("HybridItemCF_ItemCB selected")
        recommender = HybridItemCF_ItemCB.HybridItemCF_ItemCB()
        requires_icm = True

    if args.recommender == 'ItemCF':
        print("ItemCF selected")
        recommender = ItemCFKNNRecommender.ItemCFKNNRecommender()

    if args.recommender == 'SlimBPRCython_Hybrid':
        print("SlimBPRCython_Hybrid selected")
        recommender = SLIM_BPR_Cython.SLIM_BPR_Cython()

    if args.recommender == 'SlimBPR':
        print("SlimBPR")
        recommender = SLIM_BPR()

    if args.recommender == 'UserCBFKNN':
        print("UserKNNCF selected")
        recommender = UserCBFKNNRecommender.UserCBFKNNRecommender()
        requires_ucm = True

    if args.recommender == 'PureSVD':
        print("PureSVD selected")
        #recommender = PureSVDRecommender.PureSVDRecommender()

    if args.recommender == 'MF_BPR_Cython':
        print("MF_BPR_Cython selected")
        #recommender = MatrixFactorization_BPR_Cython()

    print(args)

    Runner(recommender, args.recommender, evaluate=args.eval).run(requires_icm, requires_ucm)
