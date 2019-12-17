import os, shutil

from bayes_opt import BayesianOptimization
import time

from Hybrid.Hybrid_Combo10 import Hybrid_Combo10
from Hybrid.Hybrid_Combo4 import Hybrid_Combo4
from Hybrid.Hybrid_Combo7 import Hybrid_Combo7
from Hybrid.Hybrid_Combo8 import Hybrid_Combo8
from Hybrid.Hybrid_Combo1 import Hybrid_Combo1
from Hybrid.Hybrid_Combo9 import Hybrid_Combo9
from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.Hybrid_Combo2 import Hybrid_Combo2
from Hybrid.Hybrid_Combo6 import Hybrid_Combo6
from Hybrid.Hybrid_Combo6_bis import Hybrid_Combo6_bis
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.GraphBased.P3AlphaRecommender import P3AlphaRecommender
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
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

    def step_hybrid_two(self, weight1=0, weight2=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train,  ICM_all=ICM_all, UCM_all=UCM_all, weights =[weight1, weight2],tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid_three(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all, weights=[weight1, weight2, weight3], tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid_four(self, weight1=0, weight2=0, weight3=0, weight4=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train,  ICM_all=ICM_all, UCM_all=UCM_all, weights =[weight1, weight2, weight3, weight4], tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid_6_bis(self, weight1=0, weight2=0, weight3=0, weight4=0, weight5=0, weight6=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train,  ICM_all=ICM_all, UCM_all=UCM_all, weights=[weight1, weight2, weight3, weight4, weight5, weight6], tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid_hybrid(self, weight1=0, weight2=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train,  ICM_all=ICM_all, UCM_all=UCM_all, weights=[weight1, weight2],
                                                                                       weights1=[2.28,0.07372,0.002197,0.6802],
                                                                                       weights2=[2.846, 0.0935, 0.075505, 0.6001, 2.819], tuning=True)
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
        self.recommender.fit(self.helper.URM_train, UCM_all, knn=int(weight1), shrink=int(weight2))
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_P3Alpha(self, weight1=0, weight2=0):
        start_time = time.time()
        self.recommender.fit(self.helper.URM_train, topK=int(weight1), alpha=weight2)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_p3beta(self, alpha=0, beta=0, min_rating=0, topK=0):
        start_time = time.time()
        self.recommender.fit(self.helper.URM_train, alpha=alpha, beta=beta, min_rating=min_rating, topK=int(topK))
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_PureSVD_randomSVD(self, n_components, n_iter):
        start_time = time.time()
        self.recommender.fit(self.helper.URM_train, n_components=int(n_components), n_iter=int(n_iter))
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_FunkSVD(self, epoch, num_factors, learning_rate, user_reg, item_reg):
        start_time = time.time()
        self.recommender = MatrixFactorization_FunkSVD_Cython(int(epoch), int(num_factors), learning_rate, user_reg, item_reg)
        self.recommender.fit(self.helper.URM_train)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative


if __name__ == "__main__":

    folder = os.getcwd() + "/SimilarityProduct"


    recommender = Hybrid_Combo6_bis("Combo6", UserCBFKNNRecommender())
    t = BayesianSearch(recommender, "Combo6")

    pbounds_slim = {'weight1': (250, 550), 'weight2': (100, 400)}
    pbounds_itemCB = {'weight1': (0, 200), 'weight2': (0, 200)}
    pbounds_userCB = {'weight1': (1100,1300), 'weight2': (0, 50)}

    pbounds_P3Alpha = {'weight1': (500, 1000), 'weight2': (0.5, 1.5)}
    pbounds_p3beta = {'alpha': (0, 3), 'beta': (0, 3), 'min_rating': (0, 3), 'topK': (10, 300)}

    pbounds_hybrid1 = {'weight1': (0, 3), 'weight2': (0, 3), 'weight3': (0, 3)}

    pbounds_hybrid2 = {'weight1': (0, 1), 'weight2': (0, 1), 'weight3': (0, 1), 'weight4': (0, 1)}
    pbounds_hybrid3 = {'weight1': (0.7, 1.3), 'weight2': (0.001, 0.007), 'weight3': (0.5, 3)}
    pbounds_hybrid4 = {'weight1': (1.4, 2.7), 'weight2': (1.5, 3), 'weight3': (0.0005, 0.009)}
    pbounds_hybrid5 = {'weight1': (0.005, 0.03), 'weight2': (0, 1)}
    pbounds_hybrid6 = {'weight1': (0.8, 0.95), 'weight2': (0.3, 0.45), 'weight3': (0.05, 0.065), 'weight4': (0,3)}
    pbounds_hybrid7 = {'weight1': (0, 3), 'weight2': (0, 3), 'weight3': (0, 3), 'weight4': (0, 3)}
    pbounds_hybrid8 = {'weight1': (0, 3), 'weight2': (0, 3), 'weight3': (0, 3), 'weight4': (0, 3)}

    pbounds_hybrid9_expl = {'weight1': (1.94, 1.97), 'weight2': (0.007, 0.009), 'weight3': (2.5, 3), 'weight4': (0.016, 0.019)}
    pbounds_hybrid9 = {'weight1': (0, 3), 'weight2': (0, 3), 'weight3': (0, 3), 'weight4': (0, 3)}

    pbounds_hybrid10 = {'weight1': (0, 3), 'weight2': (0, 3)}
    pbounds_hybrid10_expl = {'weight1': (0.065, 0.080), 'weight2': (2.7, 2.9)}

    # 2.65, 0.1702, 0.002764, 0.7887
    pbounds_hybrid6_bis = {'weight1': (0, 10), 'weight2': (0, 10), 'weight3': (0, 10), 'weight4': (0, 10), 'weight5': (0,10), 'weight6': (0,10)}
    pbound_random_svd = {'n_components':(100, 3000), 'n_iter':(1, 100)}
    pbound_funk_svd = {'epoch': (450,600), 'num_factors':(20,40), 'learning_rate':(0.001, 0.005), 'user_reg':(0.5, 0.9), 'item_reg':(0.1, 0.6)}

    optimizer = BayesianOptimization(
        f=t.step_hybrid_6_bis,
        pbounds=pbounds_hybrid6_bis,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=30,
        n_iter=1000,
    )
