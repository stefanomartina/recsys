import os, shutil

from bayes_opt import BayesianOptimization
import time
from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.Hybrid_Combo2 import Hybrid_Combo2
from Hybrid.Hybrid_Combo6 import Hybrid_Combo6
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender


class BayesianSearch():

    #######################################################################################
    #                             INIT CLASS BAYESIAN SEARCH                              #
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
    #                                   STEP TO MAXIMAXE                                  #
    #######################################################################################

    def step_hybrid3(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train, [weight1, weight2, weight3], ICM_all=ICM_all, UCM_all=UCM_all, tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid4(self, weight1=0):
        start_time = time.time()
        self.recommender.fit(self.helper.URM_train, weights=[weight1, 1], tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid5(self, weight1=0, weight2=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train,  ICM_all=ICM_all, UCM_all=UCM_all, weights =[weight1, weight2],
                             tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid2(self, weight1=0, weight2=0, weight3=0, weight4=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train,  ICM_all=ICM_all, UCM_all=UCM_all, weights =[weight1, weight2, weight3, weight4], tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid6(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train,  ICM_all=ICM_all, UCM_all=UCM_all, weights =[weight1, weight2, weight3],
                             tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_slim(self, weight1=0, weight2=0):
        start_time = time.time()
        self.recommender = SLIM_BPR_Cython(epochs=int(weight1), topK=int(weight2))
        self.recommender.fit(self.helper.URM_train)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_Item_CB(self, weight1=0, weight2=0):
        start_time = time.time()
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train, ICM_all, knn=int(weight1), shrink=int(weight2), tuning = False)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_User_CB(self, weight1=0, weight2=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        self.recommender.fit(self.helper.URM_train, UCM_all, knn=weight1, shrink=weight2)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

if __name__ == "__main__":

    folder = os.getcwd() + "/SimilarityProduct"

    try:
        recommender = Hybrid_Combo6("Combo 6", TopPopRecommender())
        t = BayesianSearch(recommender, "Item Content Base")

        pbounds_hybrid5 = {'weight1': (0.005, 0.03), 'weight2': (0,1)}
        pbounds_hybrid3 = {'weight1': (0.7, 1.3), 'weight2': (0.001, 0.007), 'weight3': (0.5, 3)}
        pbounds_hybrid4 = {'weight1': (0.05, 2)}
        pbounds_hybrid6 = {'weight1': (0.4, 0.45), 'weight2': (0.001, 0.005), 'weight3': (0.1, 0.5)}
        pbounds_slim = {'weight1': (250, 550), 'weight2': (100, 400)}
        pbounds_itemCB = {'weight1': (0, 200), 'weight2': (0, 200)}
        pbounds_userCB = {'weight1': (600, 900), 'weight2': (0, 50)}


        # pbounds_hybrid2 = {'weight1': (9, 10), 'weight2': (0.01, 0.04), 'weight3': (0, 0.4), 'weight4': (1.5, 1.7)}
        pbounds_hybrid2 = {'weight1': (0, 1), 'weight2': (0, 1), 'weight3': (0, 1), 'weight4': (0, 1)}

        pbounds_hybrid6 = {'weight1': (0, 3), 'weight2': (0, 3), 'weight3': (0, 3), 'weight4': (0, 3)}


        optimizer = BayesianOptimization(
            f=t.step_hybrid2,
            pbounds=pbounds_hybrid6,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(
            init_points=10,
            n_iter=1000,
            kappa=10.0
        )

    finally:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
