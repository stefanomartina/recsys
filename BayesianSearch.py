from bayes_opt import BayesianOptimization
import time
from Utils import evaluation
from Base.BaseFunction import BaseFunction
from Hybrid.HybridRecommender import HybridRecommender

class Tuner_Singles():

    def __init__(self, recommender, name):
        self.recommender = recommender
        self.name = name
        self.helper = BaseFunction()
        self.helper.get_URM()
        self.helper.split_80_20()
        self.helper.get_target_users()
        self.helper.get_UCM()
        self.helper.get_ICM()

    def step(self, weight1=0, weight2=0, weight3=0):
        start_time = time.time()
        #print("----------------------------------------")
        #print("Recommender: " + self.name + " weight1: " + str(weight1) + " weight2: " + str(weight2) + " weight3: " + str(weight3))
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, [weight1, weight2, weight3], list_ICM=list_ICM, list_UCM=list_UCM, tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

if __name__ == "__main__":
    t = Tuner_Singles(HybridRecommender("Combo3"), "Hybrid Combo3")

    pbounds = {'weight1': (0.5, 3), 'weight2': (0.001, 0.007), 'weight3': (0.5, 3)}

    optimizer = BayesianOptimization(
        f=t.step,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=1000,
    )