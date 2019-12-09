from tqdm import tqdm

from Recommenders.ContentBased import ItemCBFKNNRecommender, UserCBFKNNRecommender
from Recommenders.Combination import ItemCF_ItemCB
from Recommenders.Slim.SlimBPR.Cython import SLIM_BPR_Cython
from Recommenders.Slim.SlimElasticNet import SLIMElasticNetRecommender
from Recommenders.MatrixFactorization.PureSVD import PureSVDRecommender
from Recommenders.Collaborative import UserKNNCFRecommender, ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import RandomRecommender, TopPopRecommender
from Hybrid.HybridRecommender import HybridRecommender
from Utils import evaluation


import time
import csv
import argparse
from Base.BaseFunction import BaseFunction

FIELDS = ["user_id", "item_list"]

class Runner:

    #######################################################################################
    #                                 INSTANCE OF RUNNER                                  #
    #######################################################################################

    def __init__(self, recommender, name, evaluate=True):
        print("Evaluation: " + str(evaluate))
        self.recommender = recommender
        self.evaluate = evaluate
        self.name = name
        self.functionality = BaseFunction()

    #######################################################################################
    #                                     WRITE RESULT                                    #
    #######################################################################################

    def write_csv(self, rows, name):
        fields = FIELDS
        timestr = time.strftime("%Y-%m-%d_%H.%M.%S")
        file_path = "Results/" + name + "-" + timestr + ".csv"

        with open(file_path, 'w') as csv_file:
            csv_write_head = csv.writer(csv_file, delimiter=',')
            csv_write_head.writerow(fields)
            csv_write_content = csv.writer(csv_file, delimiter=' ')
            csv_write_content.writerows(rows)

    #######################################################################################
    #                                     RUN FITTNG                                      #
    #######################################################################################

    def fit_recommender(self, requires_icm = False, requires_ucm = False):
        print("Fitting model...")
        list_ICM = [self.functionality.ICM, self.functionality.ICM_asset, self.functionality.ICM_price]
        list_UCM = [self.functionality.UCM_age, self.functionality.UCM_region]

        if not self.evaluate:

            if requires_icm and requires_ucm:
                self.recommender.fit(self.functionality.URM_all, list_ICM, list_UCM)
            elif requires_icm:
                self.recommender.fit(self.functionality.URM_all, list_ICM)
            elif requires_ucm:
                self.recommender.fit(self.functionality.URM_all, list_UCM)
            else:
                self.recommender.fit(self.functionality.URM_all)
        else:
            self.functionality.split_80_20(0.8)
            if requires_icm and requires_ucm:
                self.recommender.fit(self.functionality.URM_train, list_ICM, list_UCM)
            elif requires_icm:
                self.recommender.fit(self.functionality.URM_train, list_ICM)
            elif requires_ucm:
                self.recommender.fit(self.functionality.URM_train, list_UCM)
            else:
                self.recommender.fit(self.functionality.URM_train)
        print("Model fitted")

    #######################################################################################
    #                                 RUN RECOMMENDATION                                  #
    #######################################################################################

    def run_recommendations(self):
        recommendations = []
        saved_tuple = []
        print("Computing recommendations...")
        for user in tqdm(self.functionality.userlist_unique):
            index = [str(user) + ","]
            recommendations.clear()

            for recommendation in self.recommender.recommend(user):
                recommendations.append(recommendation)
            saved_tuple.append(index + recommendations)
        print("Recommendations computed")
        if not self.evaluate:
            print("Printing csv...")
            self.write_csv(saved_tuple, self.name)
            print("Ended")
        return saved_tuple

    #######################################################################################
    #                                   RUN COMPUTATION                                   #
    #######################################################################################

    def run(self, requires_ucm=False, requires_icm=False):
        self.functionality.get_URM()

        if requires_icm:
            self.functionality.get_ICM()

        if requires_ucm:
            self.functionality.get_UCM()

        self.functionality.get_target_users()
        self.fit_recommender(requires_icm, requires_ucm)
        self.run_recommendations()
        if self.evaluate:
            evaluation.evaluate_algorithm(self.functionality.URM_test, self.recommender, at=10)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'top-pop',
                                                'ItemCBF', 'UserCBF', 'UserCF', 'ItemCF',
                                                'ItemCF_TopPop_Combo', 'ItemCF_ItemCB_Combo',
                                                'Slim', 'SlimElasticNet',
                                                'SlimBPRCython_Hybrid',
                                                'PureSVD', 'MF_BPR_Cython',
                                                'Hybrid'])

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

    if args.recommender == 'UserCBF':
        print("UserCBF selected")
        recommender = UserCBFKNNRecommender.UserCBFKNNRecommender()
        requires_ucm = True

    if args.recommender == 'UserCF':
        print("UserCF selected")
        recommender = UserKNNCFRecommender.UserKNNCFRecommender()

    if args.recommender == 'ItemCF':
        print("ItemCF selected")
        recommender = ItemKNNCFRecommender.ItemKNNCFRecommender()

    if args.recommender == "ItemCF_TopPop_Combo":
        print("ItemCF_TopPop_Combo selected")
        recommender = ItemCF_TopPop.ItemCF_TopPop()

    if args.recommender == "ItemCF_ItemCB_Combo":
        print("ItemCF_ItemCB_Combo selected")
        recommender = ItemCF_ItemCB.ItemCF_ItemCB()
        requires_icm = True

    if args.recommender == 'Slim':
        print("Slim selected")
        recommender = SLIM_BPR_Cython.SLIM_BPR_Cython()

    if args.recommender == 'SlimElasticNet':
        print("SlimElasticNet selected")
        recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender()

    if args.recommender == 'PureSVD':
        print("PureSVD selected")
        recommender = PureSVDRecommender.PureSVDRecommender()

    if args.recommender == "Hybrid":
        print("Hybrid")
        recommender = HybridRecommender("Combo3")
        requires_icm = True
        requires_ucm = True

    if args.recommender == 'MF_BPR_Cython':
        print("MF_BPR_Cython selected")
        #recommender = MatrixFactorization_BPR_Cython()

    print(args)
    Runner(recommender, args.recommender, evaluate=args.eval).run(requires_ucm, requires_icm)
