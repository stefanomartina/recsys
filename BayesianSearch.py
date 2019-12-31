import os, shutil

from bayes_opt import BayesianOptimization
import time

from Hybrid.Hybrid_Achille import Hybrid_Achille
from Hybrid.Hybrid_Hybrid_Combo import Hybrid_Combo10
from Hybrid.Hybrid_Combo4 import Hybrid_Combo4
from Recommenders.Slim.SlimElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.Hybrid_Achille_Tuning import Hybrid_Achille_Tuning
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.GraphBased.P3AlphaRecommender import P3AlphaRecommender
from Recommenders.NonPersonalizedRecommender.TopPopRecommender import TopPopRecommender
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from Recommenders.MatrixFactorization.ALS.ALSRecommender import AlternatingLeastSquare
from Recommenders.GraphBased.RP3BetaRecommender import RP3BetaRecommender
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

    def step_slim(self, weight3=0, weight4=0, weight5=0):
        start_time = time.time()
        self.recommender = SLIM_BPR_Cython(lambda_i=weight3, lambda_j=weight4, learning_rate=weight5)
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

    def step_all(self, H0_ICF_sh = 0, H0_ICF_tK= 0,
                   H1_UCF_sh= 0, H1_UCF_tK= 0,
                   H2_ICB_sh= 0, H2_ICB_tK= 0,
                   H3_UCB_sh=0, H3_UCB_tK=0,
                   H4_El_tK= 0,
                   H5_RP3_a= 0, H5_RP3_b= 0, H5_RP3_tK= 0,
                   H6_SL_bs= 0, H6_SL_ep= 0, H6_SL_l_i= 0, H6_SL_l_j= 0, H6_SL_l_r= 0, H6_SL_tK= 0,
                   H7_ALS_i= 0, H7_ALS_nf= 0, H7_ALS_re= 0,
                   weight1= 0, weight2= 0, weight3= 0, weight4= 0, weight5= 0, weight6= 0, weight7= 0):

        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all

        ItemCF = ItemKNNCFRecommender()
        UserCF = UserKNNCFRecommender()
        ItemCB = ItemCBFKNNRecommender()
        UserCB = UserCBFKNNRecommender()
        ElasticNet = SLIMElasticNetRecommender()
        RP3Beta = RP3BetaRecommender()
        Slim = SLIM_BPR_Cython(batch_size=int(H6_SL_bs), epochs=int(H6_SL_ep), lambda_i=H6_SL_l_i, lambda_j=H6_SL_l_j, learning_rate=H6_SL_l_r, topK=int(H6_SL_tK))
        ALS = AlternatingLeastSquare(iterations=int(H7_ALS_i), n_factors=int(H7_ALS_nf), regularization=H7_ALS_re)

        ItemCF.fit(self.helper.URM_train, knn=int(H0_ICF_tK), shrink=H0_ICF_sh)
        UserCF.fit(self.helper.URM_train, knn=int(H1_UCF_tK), shrink=H1_UCF_sh)
        ItemCB.fit(self.helper.URM_train, ICM_all, knn=int(H2_ICB_tK), shrink=H2_ICB_sh)
        UserCB.fit(self.helper.URM_train, UCM_all, knn=int(H3_UCB_tK), shrink=H3_UCB_sh)
        ElasticNet.fit(self.helper.URM_train, topK=int(H4_El_tK))
        RP3Beta.fit(self.helper.URM_train, alpha=H5_RP3_a, beta=H5_RP3_b, topK=int(H5_RP3_tK))
        Slim.fit(self.helper.URM_train)
        ALS.fit(self.helper.URM_train)

        self.recommender = Hybrid_Achille_Tuning("Hybrid_Achille_Tuning", UserCB)
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all, weights=[weight1, weight2, weight3, weight4, weight5, weight6, weight7],
                             ItemCF=ItemCF, UserCF=UserCF, ItemCB=ItemCB, ElasticNet=ElasticNet, RP3=RP3Beta, Slim=Slim, ALS=ALS)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative


if __name__ == "__main__":

    folder = os.getcwd() + "/SimilarityProduct"

    recommender = None
    t = BayesianSearch(recommender, "Achille")

    pbounds_slim = {'weight3': (0, 0.1), 'weight4': (0, 0.1), 'weight5': (0, 0.1)}
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

    # Hybrid Achille
    pbounds_hybrid_Achille = {'weight1': (4, 6), 'weight2': (0.07, 0.2), 'weight3': (0, 3), 'weight4': (4.5, 7), 'weight5': (4.5, 7), 'weight6': (0, 2), 'weight7': (0.6, 7)}

    # Tuning completo
    pbounds_all = {'H0_ICF_sh': (0, 50), 'H0_ICF_tK': (5, 800),
                   'H1_UCF_sh': (0, 50), 'H1_UCF_tK': (5, 800),
                   'H2_ICB_sh': (0, 200), 'H2_ICB_tK': (5, 800),
                   'H3_UCB_sh': (0, 50), 'H3_UCB_tK': (5, 1500),
                   'H4_El_tK': (5, 800),
                   'H5_RP3_a': (0, 3), 'H5_RP3_b': (0, 3), 'H5_RP3_tK': (5, 500),
                   'H6_SL_bs': (1, 10), 'H6_SL_ep': (400, 600), 'H6_SL_l_i': (0, 0.1), 'H6_SL_l_j': (0, 0.1), 'H6_SL_l_r': (0, 0.1), 'H6_SL_tK': (5,800),
                   'H7_ALS_i': (20, 100), 'H7_ALS_nf': (200, 600), 'H7_ALS_re': (0, 3),
                   'weight1': (0, 5), 'weight2': (0, 5), 'weight3': (0, 5), 'weight4': (0, 5), 'weight5': (0, 5), 'weight6': (0, 5), 'weight7': (0, 5)}


    optimizer = BayesianOptimization(
        f=t.step_all,
        pbounds=pbounds_all,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent


    )

    optimizer.maximize(
        init_points=30,
        n_iter=1000,
        #acq='ucb',
        #kappa = 0.1
    )
