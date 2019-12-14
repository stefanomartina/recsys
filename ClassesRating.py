from Base.BaseFunction import BaseFunction
import numpy as np

from Hybrid.Hybrid_Combo6 import Hybrid_Combo6
from Hybrid.Hybrid_Combo6_bis import Hybrid_Combo6_bis
from Hybrid.Hybrid_Combo7 import Hybrid_Combo7
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

        MAP_TopPop_per_group = []
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
        MAP_Hybrid2_per_group = []
        MAP_Hybrid6_per_group = []
        MAP_Hybrid6_bis_per_group = []
        MAP_Hybrid7_per_group = []
        MAP_Hybrid8_per_group = []


        self.profile_length = np.ediff1d(self.URM_train.indptr)
        self.blocksize = int(len(self.profile_length) * 0.05)
        self.sortedusers = np.argsort(self.profile_length)

        self.TopPop = TopPopRecommender()
        self.ItemCF = ItemKNNCFRecommender()
        self.UserCF = UserKNNCFRecommender()
        self.ItemCBF = ItemCBFKNNRecommender()
        self.UserCBF = UserCBFKNNRecommender()
        self.ItemCBF_BM25 = ItemCBFKNNRecommender()
        self.UserCBF_BM25 = UserCBFKNNRecommender()
        self.ItemCBF_TFIDF = ItemCBFKNNRecommender()
        self.UserCBF_TFIDF = UserCBFKNNRecommender()
        self.Slim = SLIM_BPR_Cython()
        self.Elastic = SLIMElasticNetRecommender()
        self.PureSVD = PureSVDRecommender()
        self.P3Alpha = P3AlphaRecommender()
        self.RP3Beta = RP3BetaRecommender()
        self.Hybrid2 = Hybrid_Combo2("HybridCombo2", TopPopRecommender())
        self.Hybrid6 = Hybrid_Combo6("HybridCombo6", TopPopRecommender())
        self.Hybrid6_bis = Hybrid_Combo6_bis("HybridCombo6_bis", UserCBFKNNRecommender())
        self.Hybrid7 = Hybrid_Combo7("HybridCombo7", TopPopRecommender())
        self.Hybrid8 = Hybrid_Combo8("HybridCombo8", TopPopRecommender())



        self.UserCBF.fit(self.URM_train, self.UCM_all)
        self.ItemCBF.fit(self.URM_train, self.ICM_all)
        self.UserCBF_BM25.fit(self.URM_train, self.UCM_all, feature_weighting="BM25")
        self.ItemCBF_BM25.fit(self.URM_train, self.ICM_all, feature_weighting="BM25")
        self.UserCBF_TFIDF.fit(self.URM_train, self.UCM_all, feature_weighting="TF-IDF")
        self.ItemCBF_TFIDF.fit(self.URM_train, self.ICM_all, feature_weighting="TF-IDF")

        """
        self.TopPop.fit(self.URM_train)
        self.ItemCF.fit(self.URM_train)
        self.UserCF.fit(self.URM_train)
        self.Slim.fit(self.URM_train)
        self.Elastic.fit(self.URM_train)
        self.PureSVD.fit(self.URM_train)
        self.P3Alpha.fit(self.URM_train)
        self.RP3Beta.fit(self.URM_train)
        self.Hybrid2.fit(self.URM_train, self.ICM_all, self.UCM_all)
        self.Hybrid7.fit(self.URM_train,  self.ICM_all,  self.UCM_all)
        self.Hybrid8.fit(self.URM_train, self.ICM_all, self.UCM_all)
        self.Hybrid6.fit(self.URM_train, self.ICM_all, self.UCM_all)
        self.Hybrid6_bis.fit(self.URM_train, self.ICM_all, self.UCM_all)
        """

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


            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.UserCBF, at=10)
            MAP_UserCBF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ItemCBF, at=10)
            MAP_ItemCBF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.UserCBF_BM25, at=10)
            MAP_UserCBF_BM25_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ItemCBF_BM25, at=10)
            MAP_ItemCBF_BM25_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.UserCBF_TFIDF, at=10)
            MAP_UserCBF_TFIDF_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.ItemCBF_TFIDF, at=10)
            MAP_ItemCBF_TFIDF_per_group.append(results)

            """
            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.TopPop, at=10)
            MAP_TopPop_per_group.append(results)
            
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
            
            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Hybrid2, at=10)
            MAP_Hybrid2_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Hybrid7, at=10)
            MAP_Hybrid7_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Hybrid8, at=10)
            MAP_Hybrid8_per_group.append(results)
            
            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Hybrid6, at=10)
            MAP_Hybrid6_per_group.append(results)

            results = evaluate_algorithm_classes(self.URM_test, users_in_group, self.Hybrid6_bis, at=10)
            MAP_Hybrid6_bis_per_group.append(results)
            """



        pyplot.plot(MAP_UserCBF_per_group, label="UserCBF")
        pyplot.plot(MAP_ItemCBF_per_group, label="ItemCBF")
        pyplot.plot(MAP_UserCBF_BM25_per_group, label="UserCBF_BM25")
        pyplot.plot(MAP_ItemCBF_BM25_per_group, label="ItemCBF_BM25")
        pyplot.plot(MAP_UserCBF_TFIDF_per_group, label="UserCBF_TFIDF")
        pyplot.plot(MAP_ItemCBF_TFIDF_per_group, label="ItemCBF_TFIDF")

        """
        pyplot.plot(MAP_ItemCF_per_group, label="ItemCF")
        pyplot.plot(MAP_UserCF_per_group, label="UserCF")
        pyplot.plot(MAP_Slim_per_group, label="Slim")
        pyplot.plot(MAP_Elastic_per_group, label="Elastic")
        pyplot.plot(MAP_P3Alpha_per_group, label="P3Alpha")
        pyplot.plot(MAP_RP3Beta_per_group, label="RP3Beta")
        pyplot.plot(MAP_PureSVD_per_group, label="PureSVD")
        pyplot.plot(MAP_TopPop_per_group, label="TopPop")       
        pyplot.plot(MAP_Hybrid2_per_group, label="Hybrid2")
        pyplot.plot(MAP_Hybrid7_per_group, label="Hybrid7")
        pyplot.plot(MAP_Hybrid8_per_group, label="Hybrid8")
        pyplot.plot(MAP_Hybrid6_per_group, label="UserCBF")
        pyplot.plot(MAP_Hybrid6_bis_per_group, label="UserCBF")
        """

        pyplot.ylabel('MAP')
        pyplot.xlabel('User Group')
        pyplot.legend()
        pyplot.show()



if __name__ == '__main__':
    stats = ClassesRating()