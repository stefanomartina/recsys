import os, shutil

from bayes_opt import BayesianOptimization
import time

from Hybrid.Hybrid_Hybrid_Combo import Hybrid_Combo10
from Hybrid.Hybrid_Combo4 import Hybrid_Combo4
from Recommenders.Slim.SlimElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.Hybrid_Combo2 import Hybrid_Combo2
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.GraphBased.P3AlphaRecommender import P3AlphaRecommender
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from Recommenders.MatrixFactorization.ALS.ALSRecommender import AlternatingLeastSquare
from Hybrid.Hybrid_user_wise import Hybrid_User_Wise
from Hybrid.Hybrid_Combo6_bis import Hybrid_Combo6_bis


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
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all, weights=[weight1, weight2, weight3, weight4, weight5, weight6], tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_hybrid_seven(self, weight1=0, weight2=0, weight3=0, weight4=0, weight5=0, weight6=0, weight7=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all,
                             weights=[weight1, weight2, weight3, weight4, weight5, weight6, weight7], tuning=True)
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

    def step_elastic(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        self.recommender.fit(self.helper.URM_train, l1_ratio=weight1, alpha = weight2, topK = int(weight3))
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_ALS(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        self.recommender = AlternatingLeastSquare(n_factors=int(weight1), regularization=weight2, iterations=int(weight3))
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

    def step_TEST(self, t1, t2, t3, t4, t5):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender = Hybrid_User_Wise("Hybrid User Wise", UserCBFKNNRecommender())
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all, thre1=t1, thre2=t2, thre3=t3, thre4=t4, thre5=t5, tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative


if __name__ == "__main__":

    folder = os.getcwd() + "/SimilarityProduct"


    recommender = SLIMElasticNetRecommender()
    t = BayesianSearch(recommender, "Elastic")

    pbounds_slim = {'weight1': (250, 550), 'weight2': (100, 400)}
    pbounds_itemCB = {'weight1': (0, 200), 'weight2': (0, 200)}
    pbounds_userCB = {'weight1': (1100,1300), 'weight2': (0, 50)}
    pbounds_P3Alpha = {'weight1': (500, 1000), 'weight2': (0.5, 1.5)}
    pbounds_p3beta = {'alpha': (0, 3), 'beta': (0, 3), 'min_rating': (0, 3), 'topK': (10, 300)}
    pbounds_ALS = {'weight1': (200, 400), 'weight2': (0.05, 0.30), 'weight3': (10, 50)}
    pbound_random_svd = {'n_components': (100, 3000), 'n_iter': (1, 100)}
    pbound_funk_svd = {'epoch': (450, 600), 'num_factors': (20, 40), 'learning_rate': (0.001, 0.005), 'user_reg': (0.5, 0.9), 'item_reg': (0.1, 0.6)}
    pbounds_elastic = {'weight1': (0, 1), 'weight2': (0, 0.1), 'weight3': (50, 300)}

    # 2.65, 0.1702, 0.002764, 0.7887
    pbounds_hybrid6_bis = {'weight1': (0, 3), 'weight2': (0, 3), 'weight3': (0, 3), 'weight4': (0, 3), 'weight5': (0,3), 'weight6': (0,3)}
    pbound_TEST = {'t1': (0, 1.5), 't2': (0.5, 1.5), 't3': (2, 2.5),'t4': (2.5, 3.5), 't5': (4, 6)}

    pbounds_hybrid_Achille = {'weight1': (0, 1), 'weight2': (0, 5), 'weight3': (0, 5), 'weight4': (0, 5), 'weight5': (0, 5), 'weight6': (0, 5), 'weight7': (0, 5)}



    optimizer = BayesianOptimization(
        f=t.step_elastic,
        pbounds=pbounds_elastic,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
        init_points=30,
        n_iter=1000,
    )
