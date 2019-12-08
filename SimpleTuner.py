import argparse
import random

import numpy as np
from Recommenders.Slim.SlimBPR.Cython import SLIM_BPR_Cython
from Hybrid.HybridRecommender import HybridRecommender
from Base.BaseFunction import BaseFunction
from Utils import evaluation


class SimpleTuner():

    #######################################################################################
    #                                  INIT CLASS TUNER                                   #
    #######################################################################################

    def __init__(self, recommender, name):
        self.recommender = recommender
        self.name = name

        self.helper = BaseFunction()
        self.helper.get_URM()
        self.helper.split_80_20()
        self.helper.get_target_users()
        self.helper.get_UCM()
        self.helper.get_ICM()

    #######################################################################################
    #                                   STEP FOR TUNING                                   #
    #######################################################################################

    def step_SlimBPR_Cython(self, epoch, topK):
        print("----------------------------------------")
        print("epoch: " + str(epoch) + " topk: " + str(topK))

        self.recommender.fit(self.helper.URM_train)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_Combo_ItemCF_ItemCB(self, topK, list_ICM, shrink, similarity):
        print("----------------------------------------")
        print("topk: " + str(topK) + " shrink: " + str(shrink) + " similarity: " + similarity)
        self.recommender.fit(self.helper.URM_train, list_ICM, topK=topK, shrink=shrink, similarity=similarity, normalize=True)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_Item_CB(self, knn, shrink):
        print("----------------------------------------")
        print("Recommender: " + self.name + " knn: " + str(knn) + " shrink: " + str(shrink))
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, list_ICM, knn=knn, shrink=shrink, tuning=True)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_User_CB(self, knn, shrink):
        print("----------------------------------------")
        print("Recommender: " + self.name + " knn: " + str(knn) + " shrink: " + str(shrink))
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        self.recommender.fit(self.helper.URM_train, list_UCM, knn=knn, shrink=shrink)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_weight_hybrid(self, list_weight):
        print("----------------------------------------")
        print("HybridCombination: " + self.name)
        print(list_weight)
        print("----------------------------------------")
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, list_ICM=list_ICM, list_UCM=list_UCM,  weights=list_weight, tuning=False)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    #######################################################################################
    #                                      RUN TUNING                                     #
    #######################################################################################

    def run_Slim(self):

        topKs = np.arange(start=10, stop=250, step=10)
        epochs = np.arange(start=200, stop=700, step=50)

        for epoch in epochs:
            for topk in topKs:
                self.recommender = SLIM_BPR_Cython.SLIM_BPR_Cython(epochs=epoch, topK=topk)
                self.step_SlimBPR_Cython(epoch, topk)

    def run_ItemCB(self):
        self.helper.split_dataset_loo()
        self.helper.get_target_users()

        topKs = np.arange(start=100, stop=500, step=50)
        shrinks = np.arange(start=0, stop=400, step=50)

        for topk in topKs:
            for shrink in shrinks:
                    self.step_Item_CB(topk, shrink)

    def run_UserCB(self):
        self.helper.split_dataset_loo()
        self.helper.get_target_users()

        topKs = np.arange(start=400, stop=600, step=10)
        shrinks = np.arange(start=0, stop=30, step=5)

        for topk in topKs:
            for shrink in shrinks:
                    self.step_User_CB(topk, shrink)

    def run_hybrid(self):
        one = 1
        weights = []

        for i in range(0, 150):
            weights.clear()

            weights.append(random.uniform(0.0015, 0.002))
            one -= weights[0]

            weights.append(one)
            one = 1
            self.step_weight_hybrid(weights)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', choices=['Combo1', 'Combo2', 'Combo3', 'Combo4', 'Combo5', 'Combo6',
                                                'Slim'])
    args = parser.parse_args()
    recommender = None

    if args.recommender == 'Combo1':
        recommender = HybridRecommender(combination='Combo1')
        SimpleTuner(recommender, args.recommender).run_hybrid()

    if args.recommender == 'Combo2':
        recommender = HybridRecommender(combination='Combo2')
        SimpleTuner(recommender, args.recommender).run_hybrid()

    if args.recommender == 'Combo3':
        recommender = HybridRecommender(combination='Combo3')
        SimpleTuner(recommender, args.recommender).run_hybrid()

    if args.recommender == 'Combo4':
        recommender = HybridRecommender(combination='Combo4')
        SimpleTuner(recommender, args.recommender).run_hybrid()

    if args.recommender == 'Combo5':
        recommender = HybridRecommender(combination='Combo5')
        SimpleTuner(recommender, args.recommender).run_hybrid()

    if args.recommender == 'Combo6':
        recommender = HybridRecommender(combination='Combo6')
        SimpleTuner(recommender, args.recommender).run_hybrid()

    if args.recommender == 'Slim':
        SimpleTuner(recommender, args.recommender).run_Slim()
