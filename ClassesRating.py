from Base.BaseFunction import BaseFunction
import numpy as np
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Utils.evaluation import evaluate_algorithm
import matplotlib.pyplot as pyplot


class ClassesRating:
    def __init__(self):
        helper = BaseFunction()
        self.URM = None
        self.URM_train = None
        self.URM_test = None
        self.URM = helper.get_URM()
        helper.split_80_20()
        self.URM_train, self.URM_test = helper.URM_train, helper.URM_test
        helper.get_ICM()
        helper.get_UCM()
        self.list_ICM, self.list_UCM = [helper.ICM, helper.ICM_asset, helper.ICM_price], [helper.UCM_age, helper.UCM_region]

        MAP_TopPop_per_group = []
        MAP_ItemCF_per_group = []
        MAP_UserCF_per_group = []
        MAP_ItemCBF_per_group = []
        MAP_UserCBF_per_group = []
        MAP_Slim_per_group = []

        self.profile_length = np.ediff1d(self.URM_train.indptr)
        self.blocksize = int(len(self.profile_length) * 0.05)
        self.sortedusers = np.argsort(self.profile_length)

        self.TopPop = TopPopRecommender().fit(self.URM_train)
        self.ItemCF = ItemKNNCFRecommender().fit(self.URM_train)
        self.UserCF = UserKNNCFRecommender().fit(self.URM_train)
        self.ItemCBF = ItemCBFKNNRecommender().fit(self.URM_train, self.list_ICM)
        self.UserCBF = UserCBFKNNRecommender().fit(self.URM_train, self.list_UCM)
        self.Slim = SLIM_BPR_Cython().fit(self.URM_train)

        for group_id in range(0, 10):
            start_pos = group_id * self.blocksize
            end_pos = min((group_id + 1) * self.blocksize, len(self.profile_length))

            users_in_group = self.sortedusers[start_pos:end_pos]

            users_in_group_p_len = self.profile_length[users_in_group]

            print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                          users_in_group_p_len.mean(),
                                                                          users_in_group_p_len.min(),
                                                                          users_in_group_p_len.max()))
            users_not_in_group_flag = np.isin(self.sortedusers, users_in_group, invert=True)
            users_not_in_group = self.sortedusers[users_not_in_group_flag]

            results = evaluate_algorithm(self.URM_test, self.TopPop, at=10)
            MAP_TopPop_per_group.append(results)

            results = evaluate_algorithm(self.URM_test, self.ItemCF, at=10)
            MAP_ItemCF_per_group.append(results)

            results = evaluate_algorithm(self.URM_test, self.UserCF, at=10)
            MAP_UserCF_per_group.append(results)

            results = evaluate_algorithm(self.URM_test, self.ItemCBF, at=10)
            MAP_ItemCBF_per_group.append(results)

            results = evaluate_algorithm(self.URM_test, self.UserCBF, at=10)
            MAP_UserCBF_per_group.append(results)

            results = evaluate_algorithm(self.URM_test, self.Slim, at=10)
            MAP_Slim_per_group.append(results)

        pyplot.plot(MAP_TopPop_per_group, label="TopPop")
        pyplot.plot(MAP_ItemCF_per_group, label="ItemCF")
        pyplot.plot(MAP_UserCF_per_group, label="UserCF")
        pyplot.plot(MAP_ItemCBF_per_group, label="ItemCBF")
        pyplot.plot(MAP_UserCBF_per_group, label="UserCBF")
        pyplot.plot(MAP_Slim_per_group, label="Slim")
        pyplot.ylabel('MAP')
        pyplot.xlabel('User Group')
        pyplot.legend()
        pyplot.show()



if __name__ == '__main__':
    stats = ClassesRating()