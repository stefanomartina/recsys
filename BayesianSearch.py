"""
This class is an implementation of Bayesian research regarding the parameters considered 'tunable' of the
individual algorithms.
"""

import time
import argparse
from bayes_opt import BayesianOptimization

from Hybrid.Hybrid_Achille import Hybrid_Achille
from Hybrid.Hybrid_Combo3 import Hybrid_Combo3
from Hybrid.Hybrid_Combo4 import Hybrid_Combo4
from Recommenders.Slim.SlimElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.Hybrid_Achille_Tuning_All import Hybrid_Achille_Tuning
from Hybrid.Hybrid_Combo6 import Hybrid_Combo6
from Hybrid.Hybrid_Achille_Tuning_FallBack import Hybrid_Achille_Tuning_FallBack
from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.ContentBased.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from Recommenders.ContentBased.UserCBFKNNRecommender import UserCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython
from Recommenders.MatrixFactorization.ALS.ALSRecommender import AlternatingLeastSquare
from Recommenders.MatrixFactorization.PureSVD.PureSVDRecommender import PureSVDRecommender
from Recommenders.GraphBased.RP3BetaRecommender import RP3BetaRecommender
from Recommenders.GraphBased.P3AlphaRecommender import P3AlphaRecommender
from Hybrid.Hybrid_UW import Hybrid_User_Wise


class BayesianSearch:

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
        self.optimazer = None

    def instanziate_optimazer(self, bayesian_method_call, pbounds):
        optimizer = BayesianOptimization(
            f=bayesian_method_call,
            pbounds=pbounds,
            verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        )

        optimizer.maximize(
            init_points=30,
            n_iter=1000,
            acq='ucb',
            kappa=0.1
        )

    #######################################################################################
    #                                   STEP TO MAXIMAXE                                  #
    #######################################################################################

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

    def step_hybrid_six(self, weight1=0, weight2=0, weight3=0, weight4=0, weight5=0, weight6=0):
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

    def step_fallBack_Hybrid(self, weight1=0, weight2=0):
        start_time = time.time()
        UCM_all = self.helper.UCM_all
        ICM_all = self.helper.ICM_all
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all,
                             weights_fallback=[int(weight1), int(weight2)], tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def step_slim(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        self.recommender = SLIM_BPR_Cython(lambda_i=weight1, lambda_j=weight2, learning_rate=weight3)
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

    def step_RP3Beta(self, alpha=0, beta=0, min_rating=0, topK=0):
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
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all, thre1=t1, thre2=t2, thre3=t3,
                             thre4=t4, thre5=t5, tuning=True)
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

        self.recommender = Hybrid_Achille_Tuning("Hybrid_Achille_Tuning_All", UserCB)
        self.recommender.fit(self.helper.URM_train, ICM_all=ICM_all, UCM_all=UCM_all, weights=[weight1, weight2, weight3, weight4, weight5, weight6, weight7],
                             ItemCF=ItemCF, UserCF=UserCF, ItemCB=ItemCB, ElasticNet=ElasticNet, RP3=RP3Beta, Slim=Slim, ALS=ALS)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('research_type', choices=['C3', 'C4', 'C6', 'UW', 'Achille', 'Achille_TF', 'Achille_TA',
                                                  'Search_Slim', 'Search_Elastic', 'Search_ItemCB', 'Search_UserCB',
                                                  'Search_P3Alpha', 'Search_RP3Beta', 'Search_ALS', 'Search_PureSVD',
                                                  'Search_FunkSVD'])
    args = parser.parse_args()
    recommender = None
    name_search = None

    if args.research_type == 'C3':
        recommender = Hybrid_Combo3("Hybrid_Combo3", UserCBFKNNRecommender())
        name_search = "Hybrid_Combo3"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (0, 0.1), 'weight2': (0, 0.1), 'weight3': (0, 0.1), 'weight4': (0, 0.1)}
        t.instanziate_optimazer(t.step_hybrid_four, pbounds)

    if args.research_type == 'C4':
        recommender = Hybrid_Combo4("Hybrid_Combo4", UserCBFKNNRecommender())
        name_search = "Hybrid_Combo4"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (0, 0.1), 'weight2': (0, 0.1), 'weight3': (0, 0.1)}
        t.instanziate_optimazer(t.step_hybrid_three, pbounds)

    if args.research_type == 'C6':
        recommender = Hybrid_Combo6("Hybrid_Combo6", UserCBFKNNRecommender())
        name_search = "Hybrid_Combo6"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (0, 3), 'weight2': (0, 3), 'weight3': (0, 3), 'weight4': (0, 3), 'weight5': (0, 3),
                   'weight6': (0, 3)}
        t.instanziate_optimazer(t.step_hybrid_six, pbounds)

    if args.research_type == 'UW':
        recommender = None
        name_search = "Hybrid_UW"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'t1': (0, 1.5), 't2': (0.5, 1.5), 't3': (2, 2.5),'t4': (2.5, 3.5), 't5': (4, 6)}
        t.instanziate_optimazer(t.step_hybrid_six, pbounds)

    if args.research_type == 'Achille':
        recommender = Hybrid_Achille("Hybrid_Achille", UserCBFKNNRecommender())
        name_search = "Hybrid_Achille"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (0, 10), 'weight2': (0, 10), 'weight3': (0, 10), 'weight4': (0, 10), 'weight5': (0, 10),
                   'weight6': (0, 10), 'weight7': (0, 10)}
        t.instanziate_optimazer(t.step_hybrid_seven, pbounds)

    if args.research_type == 'Achille_TF':
        recommender = Hybrid_Achille("Hybrid_Achille_Tuning_FallBack", UserCBFKNNRecommender())
        name_search = "Hybrid_Achille_Tuning_FallBack"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (0, 10), 'weight2': (0, 10)}
        t.instanziate_optimazer(t.step_fallBack_Hybrid, pbounds)

    if args.research_type == 'Achille_TA':
        recommender = Hybrid_Achille("Hybrid_Achille_Tuning_All", None)
        name_search = "Hybrid_Achille_Tuning_All"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'H0_ICF_sh': (0, 50), 'H0_ICF_tK': (5, 800),
                   'H1_UCF_sh': (0, 50), 'H1_UCF_tK': (5, 800),
                   'H2_ICB_sh': (0, 200), 'H2_ICB_tK': (5, 800),
                   'H3_UCB_sh': (0, 50), 'H3_UCB_tK': (5, 1500),
                   'H4_El_tK': (5, 800),
                   'H5_RP3_a': (0, 3), 'H5_RP3_b': (0, 3), 'H5_RP3_tK': (5, 500),
                   'H6_SL_bs': (1, 10), 'H6_SL_ep': (400, 600), 'H6_SL_l_i': (0, 0.1), 'H6_SL_l_j': (0, 0.1),
                   'H6_SL_l_r': (0, 0.1), 'H6_SL_tK': (5, 800),
                   'H7_ALS_i': (20, 100), 'H7_ALS_nf': (200, 600), 'H7_ALS_re': (0, 3),
                   'weight1': (0, 5), 'weight2': (0, 5), 'weight3': (0, 5), 'weight4': (0, 5), 'weight5': (0, 5),
                   'weight6': (0, 5), 'weight7': (0, 5)}

        t.instanziate_optimazer(t.step_all, pbounds)

    if args.research_type == 'Search_Slim':
        recommender = None
        name_search = "Slim_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (0, 0.1), 'weight2': (0, 0.1), 'weight3': (0, 0.1)}
        t.instanziate_optimazer(t.step_slim, pbounds)

    if args.research_type == 'Search_Elastic':
        recommender = SLIMElasticNetRecommender()
        name_search = "Slim_Elastic_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (0, 1), 'weight2': (0, 0.1), 'weight3': (50, 300)}
        t.instanziate_optimazer(t.step_elastic, pbounds)

    if args.research_type == 'Search_ItemCB':
        recommender = ItemCBFKNNRecommender()
        name_search = "Item_CB_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (10, 20), 'weight2': (20, 60)}
        t.instanziate_optimazer(t.step_Item_CB, pbounds)

    if args.research_type == 'Search_UserCB':
        recommender = UserCBFKNNRecommender()
        name_search = "User_CB_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (1600, 1900), 'weight2': (50, 80)}
        t.instanziate_optimazer(t.step_User_CB, pbounds)

    if args.research_type == 'Search_P3Alpha':
        recommender = P3AlphaRecommender()
        name_search = "P3Alpha_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (500, 1000), 'weight2': (0.5, 1.5)}
        t.instanziate_optimazer(t.step_P3Alpha, pbounds)

    if args.research_type == 'Search_RP3Beta':
        recommender = RP3BetaRecommender()
        name_search = "RP3Beta_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'alpha': (0, 3), 'beta': (0, 3), 'min_rating': (0, 3), 'topK': (10, 300)}
        t.instanziate_optimazer(t.step_RP3Beta, pbounds)

    if args.research_type == 'Search_ALS':
        recommender = None
        name_search = "ALS_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'weight1': (200, 400), 'weight2': (0.05, 0.30), 'weight3': (10, 50)}
        t.instanziate_optimazer(t.step_ALS, pbounds)

    if args.research_type == 'Search_PureSVD':
        recommender = PureSVDRecommender()
        name_search = "PureSVD_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'n_components': (100, 3000), 'n_iter': (1, 100)}
        t.instanziate_optimazer(t.step_PureSVD_randomSVD, pbounds)

    if args.research_type == 'Search_FunkSVD':
        recommender = None
        name_search = "FunkSVD_Recommender"
        t = BayesianSearch(recommender, name_search)
        pbounds = {'epoch': (450, 600), 'num_factors': (20, 40), 'learning_rate': (0.001, 0.005),
                   'user_reg': (0.5, 0.9), 'item_reg': (0.1, 0.6)}
        t.instanziate_optimazer(t.step_FunkSVD, pbounds)