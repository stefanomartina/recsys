from Base.BaseFunction import BaseFunction
import numpy as np
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Utils.evaluation import evaluate_algorithm, evaluate_algorithm_classes
import matplotlib.pyplot as pyplot


class ClassesRating:
    def __init__(self):
        self.helper = BaseFunction()
        self.URM = None
        self.URM_train = None
        self.URM_test = None
        self.URM = self.helper.get_URM()
        self.helper.split_80_20()
        self.URM_train, self.URM_test = self.helper.URM_train, self.helper.URM_test
        self.helper.get_ICM()
        self.helper.get_UCM()
        self.helper.get_target_users()
        self.ICM_all = self.helper.ICM_all
        self.UCM_all = self.helper.UCM_all
        self.initial_target_user = self.helper.userlist_unique

        MAP_TopPop_per_group = []
        MAP_ItemCF_per_group = []
        MAP_UserCF_per_group = []
        MAP_ItemCBF_per_group = []
        MAP_UserCBF_per_group = []
        MAP_Slim_per_group = []

        self.profile_length = np.ediff1d(self.URM_train.indptr)
        self.blocksize = int(len(self.profile_length) * 0.05)
        self.sortedusers = np.argsort(self.profile_length)

        self.TopPop = TopPopRecommender()
        self.ItemCF = ItemKNNCFRecommender()
        self.UserCF = UserKNNCFRecommender()
        self.ItemCBF = ItemCBFKNNRecommender()
        self.UserCBF = UserCBFKNNRecommender()
        self.Slim = SLIM_BPR_Cython()

        self.TopPop.fit(self.URM_train)
        self.ItemCF.fit(self.URM_train)
        self.UserCF.fit(self.URM_train)
        self.ItemCBF.fit(self.URM_train, self.ICM_all)
        self.UserCBF.fit(self.URM_train, self.UCM_all)
        self.Slim.fit(self.URM_train)

        for group_id in range(0, 20):
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

            users_in_group = list(set(self.initial_target_user) - set(list(users_not_in_group)))

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.TopPop, at=10)
            MAP_TopPop_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ItemCF, at=10)
            MAP_ItemCF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.UserCF, at=10)
            MAP_UserCF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ItemCBF, at=10)
            MAP_ItemCBF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.UserCBF, at=10)
            MAP_UserCBF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Slim, at=10)
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