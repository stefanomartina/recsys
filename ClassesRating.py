from Base.BaseFunction import BaseFunction
import numpy as np

from Hybrid.Hybrid_CB import Hybrid_CB
from Hybrid.Hybrid_Combo4 import Hybrid_Combo4
from Hybrid.Hybrid_Combo6_bis import Hybrid_Combo6_bis
from Hybrid.Hybrid_Combo8 import Hybrid_Combo8
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.MatrixFactorization.PureSVD.PureSVDRecommender import PureSVDRecommender
from Recommenders.GraphBased.P3AlphaRecommender import P3AlphaRecommender
from Recommenders.GraphBased.RP3BetaRecommender import RP3BetaRecommender
from Recommenders.Slim.SlimElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Recommenders.MatrixFactorization.ALS.ALSRecommender import AlternatingLeastSquare
from Hybrid.Hybrid_Combo2 import Hybrid_Combo2
from Utils.evaluation import evaluate_algorithm_classes
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

        MAP_ItemCF_per_group = []
        MAP_UserCF_per_group = []
        MAP_ItemCBF_per_group = []
        MAP_UserCBF_per_group = []
        MAP_ItemCBF_BM25_per_group = []
        MAP_UserCBF_BM25_per_group = []
        MAP_ItemCBF_TFIDF_per_group = []
        MAP_UserCBF_TFIDF_per_group = []
        MAP_Slim_per_group = []
        MAP_Elastic_per_group = []
        MAP_PureSVD_per_group = []
        MAP_P3Alpha_per_group = []
        MAP_RP3Beta_per_group = []
        MAP_ALS_per_group = []
        MAP_Hybrid2_per_group = []
        MAP_Hybrid6_per_group = []
        MAP_H6_bis_per_group = []
        MAP_Hybrid7_per_group = []
        MAP_Hybrid8_per_group = []
        MAP_HybridCB_per_group = []


        self.profile_length = np.ediff1d(self.URM_train.indptr)
        self.blocksize = int(len(self.profile_length) * 0.05)
        self.sortedusers = np.argsort(self.profile_length)

        self.ItemCF = ItemKNNCFRecommender()
        self.UserCF = UserKNNCFRecommender()
        self.ItemCBF = ItemCBFKNNRecommender()
        self.UserCBF = UserCBFKNNRecommender()
        self.Slim = SLIM_BPR_Cython()
        self.Elastic = SLIMElasticNetRecommender()
        self.PureSVD = PureSVDRecommender()
        self.P3Alpha = P3AlphaRecommender()
        self.RP3Beta = RP3BetaRecommender()
        self.ALS = AlternatingLeastSquare()
        self.H6_bis = Hybrid_Combo6_bis("Combo6_bis", UserCBFKNNRecommender())

        self.ItemCBF.fit(self.URM_train, self.ICM_all, tuning=True, similarity_path="/SimilarityProduct/ItemCBF_similarity.npz")
        self.UserCBF.fit(self.URM_train, self.UCM_all, tuning=True, similarity_path="/SimilarityProduct/UserCBF_similarity.npz")
        self.ItemCF.fit(self.URM_train, tuning=True, similarity_path="/SimilarityProduct/ItemCF_similarity.npz")
        self.UserCF.fit(self.URM_train, tuning=True, similarity_path="/SimilarityProduct/UserCF_similarity.npz")
        self.Slim.fit(self.URM_train, tuning=True, similarity_path="/SimilarityProduct/Slim_similarity.npz")
        self.Elastic.fit(self.URM_train, tuning=True, similarity_path="/SimilarityProduct/Elastic_similarity.npz")
        self.PureSVD.fit(self.URM_train)
        self.P3Alpha.fit(self.URM_train, tuning=True, similarity_path="/SimilarityProduct/P3Aplha_similarity.npz")
        self.RP3Beta.fit(self.URM_train, tuning=True, similarity_path="/SimilarityProduct/RP3Beta_similarity.npz")
        self.ALS.fit(self.URM_train)
        self.H6_bis.fit(self.URM_train, self.ICM_all, self.UCM_all, tuning=True)

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


            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ItemCBF, at=10)
            MAP_ItemCBF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ItemCF, at=10)
            MAP_ItemCF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.UserCF, at=10)
            MAP_UserCF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Slim, at=10)
            MAP_Slim_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Elastic, at=10)
            MAP_Elastic_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.PureSVD, at=10)
            MAP_PureSVD_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.P3Alpha, at=10)
            MAP_P3Alpha_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.RP3Beta, at=10)
            MAP_RP3Beta_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.UserCBF, at=10)
            MAP_UserCBF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ALS, at=10)
            MAP_ALS_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.H6_bis, at=10)
            MAP_H6_bis_per_group.append(results)


        pyplot.plot(MAP_UserCBF_per_group, label="UserCBF")
        pyplot.plot(MAP_ItemCBF_per_group, label="ItemCBF")
        pyplot.plot(MAP_ItemCF_per_group, label="ItemCF")
        pyplot.plot(MAP_UserCF_per_group, label="UserCF")
        pyplot.plot(MAP_Slim_per_group, label="Slim")
        pyplot.plot(MAP_Elastic_per_group, label="Elastic")
        pyplot.plot(MAP_P3Alpha_per_group, label="P3Alpha")
        pyplot.plot(MAP_RP3Beta_per_group, label="RP3Beta")
        pyplot.plot(MAP_PureSVD_per_group, label="PureSVD")
        pyplot.plot(MAP_ALS_per_group, label="ALS")
        pyplot.plot(MAP_H6_bis_per_group, label="H6_bis")

        pyplot.xlabel('User Group')
        pyplot.ylabel('MAP')
        pyplot.xticks(np.arange(0, 20, 1))
        pyplot.grid(b=True, axis='both', color='firebrick', linestyle='--', linewidth=0.5)
        pyplot.legend(loc='lower right')
        pyplot.show()


class LongTail:
    def __init__(self):
        self.helper = BaseFunction()
        self.URM = None
        self.URM_train = None
        self.URM_test = None
        self.helper.get_URM()
        self.helper.split_80_20()
        self.URM_train, self.URM_test = self.helper.URM_train, self.helper.URM_test

        cont = 0

        itemPopularity = (self.URM_train > 0).sum(axis=0)
        self.itemPopularity = list(np.array(itemPopularity).squeeze())
        self.itemPopularity.sort(reverse=True)

        for i in range(0, len(self.itemPopularity)):
            if self.itemPopularity[i] < 5:
                cont += 1
        print(cont)


        # 14959 hanno meno di 20 ratings
        # 12675 hanno meno di 10 ratings
        # 9829 hanno meno di 5 ratings


        """
        pyplot.plot(self.itemPopularity, color='red')
        pyplot.ylabel('Pupularity')
        pyplot.xlabel('Items')
        pyplot.legend()
        pyplot.show()
        """



if __name__ == '__main__':
    stats = ClassesRating()