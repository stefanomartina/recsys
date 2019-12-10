import os, shutil

from bayes_opt import BayesianOptimization
import time
from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.Hybrid_Combo2 import Hybrid_Combo2
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
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, [weight1, weight2, weight3], list_ICM=list_ICM, list_UCM=list_UCM, tuning=True)
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
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train,  list_ICM=list_ICM, list_UCM=list_UCM,weights =[weight1, weight2],
                             tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid2(self, weight1=0, weight2=0, weight3=0, weight4=0):
        start_time = time.time()
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train,  list_ICM=list_ICM, list_UCM=list_UCM,weights =[weight1, weight2, weight3, weight4],
                             tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid6(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train,  list_ICM=list_ICM, list_UCM=list_UCM,weights =[weight1, weight2, weight3],
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
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, list_ICM, knn=weight1, shrink=weight2)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_User_CB(self, weight1=0, weight2=0):
        start_time = time.time()
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        self.recommender.fit(self.helper.URM_train, list_UCM, knn=weight1, shrink=weight2)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

if __name__ == "__main__":

    folder = os.getcwd() + "/SimilarityProduct"

    try:
        recommender = Hybrid_Combo2("Combo2", TopPopRecommender())
        t = BayesianSearch(recommender, "Hybrid: Combo 2")

        pbounds_hybrid5 = {'weight1': (0.005, 0.03), 'weight2': (0,1)}
        pbounds_hybrid3 = {'weight1': (0.7, 1.3), 'weight2': (0.001, 0.007), 'weight3': (0.5, 3)}

        pbounds_hybrid2 = {'weight1': (0, 10), 'weight2': (0, 10), 'weight3': (0, 10), 'weight4': (0, 10)}

        pbounds_hybrid4 = {'weight1': (0.05, 2)}


        pbounds_hybrid6 = {'weight1': (0.4, 0.45), 'weight2': (0.001, 0.005), 'weight3': (0.1, 0.5)}
        pbounds_slim = {'weight1': (250, 550), 'weight2': (100, 400)}
        pbounds_itemCB = {'weight1': (600, 900), 'weight2': (0, 50)}
        pbounds_userCB = {'weight1': (600, 900), 'weight2': (0, 50)}



        optimizer = BayesianOptimization(
            f=t.step_hybrid2,
            pbounds=pbounds_hybrid2,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(
            init_points=10,
            n_iter=1000,
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
