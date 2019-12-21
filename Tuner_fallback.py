import os, shutil

from bayes_opt import BayesianOptimization
import time

from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.Hybrid_Combo6_bis import Hybrid_Combo6_bis
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython


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

    def step(self, w1, w2):
        start_time = time.time()
        UCM_all = self.helper.UCM_all

        self.recommender.fit(self.helper.URM_train, UCM_all=UCM_all, tuning=True, w1=w1, w2=w2)

        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative


if __name__ == "__main__":

    folder = os.getcwd() + "/SimilarityProduct"


    recommender = UserCBFKNNRecommender()
    t = BayesianSearch(recommender, "UserCBF")

    pbounds = {'w1': (0, 3), 'w2': (0, 3)}


    optimizer = BayesianOptimization(
        f=t.step,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=30,
        n_iter=1000,
    )
