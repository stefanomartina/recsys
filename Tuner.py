import random
from _decimal import Decimal

import numpy as np
import time
from Recommenders.Combination.ItemCF_ItemCB import ItemCF_ItemCB
from Recommenders.Slim.SlimBPR.Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import ItemCBFKNNRecommender, UserCBFKNNRecommender
from Hybrid.HybridRecommender import HybridRecommender
from Base.BaseFunction import BaseFunction
from Utils import evaluation


class Tuner():

    def __init__(self, recommender, name):
        self.recommender = recommender
        self.name = name
        self.helper = BaseFunction()
        self.helper.get_URM()
        self.helper.get_UCM()
        self.helper.get_ICM()

    def step_HybridItemCF_ItemCB(self, topK, list_ICM, shrink, similarity):
        print("----------------------------------------")
        print("topk: " + str(topK) + " shrink: " + str(shrink) + " similarity: " + similarity)
        self.recommender.fit(self.helper.URM_train, list_ICM, topK=topK, shrink=shrink, similarity=similarity, normalize=True)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_SlimBPR_Cython(self, topK, learning_rate, L1, L2, sgd_mode, gamma=0.995, beta1=0.9, beta2=0.999):
        print("----------------------------------------")
        print("topk: " + str(topK) + " learning_rate: " + str(learning_rate) + " L1: " + str(L1) + " L2: " + str(L2) + " beta1: " + str(beta1) + " beta2: " + str(beta2) +
             " gamma: " + str(gamma) + " sgd_mode: " + sgd_mode)

        self.recommender.fit(self.helper.URM_train, topK=topK, learning_rate=learning_rate, lambda_i = L1, lambda_j = L2, beta_1 = beta1, beta_2 = beta2, gamma = gamma, sgd_mode = sgd_mode)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_Item_CB(self, knn, shrink):
        print("----------------------------------------")
        print("Recommender: " + self.name + " knn: " + str(knn) + " shrink: " + str(shrink))
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, list_ICM, knn=knn, shrink=shrink)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_User_CB(self, knn, shrink):
        print("----------------------------------------")
        print("Recommender: " + self.name + " knn: " + str(knn) + " shrink: " + str(shrink))
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        self.recommender.fit(self.helper.URM_train, list_UCM, knn=knn, shrink=shrink)
        evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_weight(self, list_weight):
        start_time = time.time()
        print("----------------------------------------")
        # print("Recommender: " + self.name + " First weight: " + str(list_weight[0]) +
        #     " Second_weight: " + str(list_weight[1]) + " Third_weight: " + str(list_weight[2]) + " Fourth_weight: " + str(list_weight[3]))

        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, list_ICM, list_UCM,list_weight, tuning=True)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def run_Slim(self):
        self.helper.split_dataset_loo()
        self.helper.get_target_users()

        topKs = np.arange(start=10, stop = 200, step = 10)
        learning_rates = [1e-2, 1e-3, 1e-4]
        lambda_is = [0.0001, 0.0002]
        lambda_js = [0.0001, 0.0002]
        beta_1s = [0.899, 0.9]
        beta_2s = [0.9, 0.999]
        gammas = np.arange(start = 0.99, stop = 0.999, step = 0.001)
        sgd_modes = ["adagrad", "rmsprop", "adam", "sgd"]

        for topk in topKs:
            for learning_rate in learning_rates:
                for lambda_i in lambda_is:
                    for lambda_j in lambda_js:
                            for sgd_mode in sgd_modes:
                                if sgd_mode == "rmsprop":
                                    for gamma in gammas:
                                        self.step_SlimBPR_Cython(topk, learning_rate, lambda_i, lambda_j, sgd_mode, gamma)
                                elif sgd_mode == "adam":
                                    for beta_1 in beta_1s:
                                        for beta_2 in beta_2s:
                                            self.step_SlimBPR_Cython(topk, learning_rate, lambda_i, lambda_j, sgd_mode, beta_1, beta_2)

    def run_Item(self):
        self.helper.split_dataset_loo()
        self.helper.get_target_users()

        topKs = np.arange(start=100, stop=500, step=50)
        shrinks = np.arange(start=0, stop=400, step=50)

        for topk in topKs:
            for shrink in shrinks:
                    self.step_Item_CB(topk, shrink)

    def run_User(self):
        self.helper.split_dataset_loo()
        self.helper.get_target_users()

        topKs = np.arange(start=10, stop=210, step=20)
        shrinks = np.arange(start=10, stop=210, step=20)

        for topk in topKs:
            for shrink in shrinks:
                    self.step_User_CB(topk, shrink)

    def run_hybrid(self):
        self.helper.split_dataset_loo()

        self.helper.get_target_users()

        one = 1
        weights = []

        for i in range(0, 30):
            weights.clear()
            #self.step_weight(np.random.dirichlet(np.ones(4), size=1))
            weights.append(random.uniform(0, 0.2))
            one -= weights[0]

            weights.append(random.uniform(0, 0.2))
            one -= weights[1]

            weights.append(random.uniform(0.1, 0.2))
            one -= weights[2]

            weights.append(one)
            one = 1
            self.step_weight(weights)

    def random_pop(self):
        weights = []
        for i in range(self.pop_size):
            w1 = random.uniform(0, 0.2)
            w2 = random.uniform(0, 0.2)
            w3 = random.uniform(0.3, 0.5)
            res = 1 - w1 - w2 - w3
            w4 = random.uniform(0, res)
            line = [w1, w2, w3, w4]
            weights.append(line)

        return weights

    def evaluate_pop(self):
        appo = []
        for chromosome in self.pop:
            res = self.evaluate_chromosome(chromosome)
            appo.append(res)
        return appo
        # return [self.evaluate_chromosome(chromosome) for chromosome in self.pop]

    def evaluate_chromosome(self, chromosome):
        return self.step_weight(chromosome)

    def select_parents(self):
        max_score = max(self.pop_scores)
        adj_scores = [max_score + 1 - score for score in self.pop_scores]
        tot_score = sum(adj_scores)
        probs = [p/tot_score for p in adj_scores]
        parents = [self.pop[i] for i in np.random.choice(len(self.pop)-1, p = probs)]

        return parents

    def generate_offspring(self, p1, p2, b1, b2):
        c1, c2 = p1[:], p2[:]

        size = len(c1)
        offspring = np.empty((size), dtype='object')

        for i in range(b1, b2 + 1):
            offspring[i] = c1[i]

        for i in range(b2+1, size):
            offspring[i] = c2[i]

        return offspring

    def crossover(self, parents):
        b1, b2 = np.sort(np.random.randint(0, len(parents[0]), size=2))

        offspring1 = self.generate_offspring(parents[0], parents[1], b1, b2)
        offspring2 = self.generate_offspring(parents[1], parents[0], b1, b2)

        offspring1 = self.mutation(offspring1)
        offspring2 = self.mutation(offspring2)

        return offspring1, offspring2

    def mutation(self, offspring):
        offspring += int(random.randrange(0, 0.1))
        return offspring

    def elitism(self, new_pop):
        els = self.pop[:]
        score_c = self.pop_scores[:]

        for _ in range(4):
            index = np.argmin(score_c)
            score_c.pop(index)
            new_pop.append(els.pop(index))



    def run_hybrid_hill_climbing(self, max = 1000, pop_size=5, p_mutation=0.1):
        self.helper.split_dataset_loo()
        self.helper.get_target_users()
        self.pop_size = pop_size
        self.p_mutation = p_mutation

        self.pop = self.random_pop()
        self.pop_scores = self.evaluate_pop()

        for i in range(max):
            new_pop = []
            self.elitism(new_pop)

            while(len(new_pop) < len(self.pop)):
                parents = self.select_parents()
                off1, off2 = self.crossover(parents)
                new_pop.append(off1)
                new_pop.append(off2)

            self.pop_scores = new_pop
            self.pop_scores = self.evaluate_pop()

            print("best score: %i" % np.min(self.pop_scores))
            print("best res: %i" % np.argmin(self.pop_scores))


if __name__ == "__main__":
    recommender = HybridRecommender()
    Tuner(recommender, "Item Content Base").run_hybrid_hill_climbing()
