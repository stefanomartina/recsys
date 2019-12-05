import random
from _decimal import Decimal

import numpy as np
import time
from Recommenders.Combination.ItemCF_ItemCB import ItemCF_ItemCB
from Recommenders.Slim.SlimBPR.Cython import SLIM_BPR_Cython
from Recommenders.ContentBased import ItemCBFKNNRecommender, UserCBFKNNRecommender
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Hybrid.HybridRecommender import HybridRecommender
from Base.BaseFunction import BaseFunction
from Utils import evaluation


class Tuner_Singles():

    def __init__(self, recommender, name):
        self.recommender = recommender
        self.name = name
        self.helper = BaseFunction()
        self.helper.get_URM()
        self.helper.split_80_20()
        self.helper.get_target_users()

    def step_User_CB(self, knn, shrink):
        start_time = time.time()
        print("----------------------------------------")
        print("Recommender: " + self.name + " knn: " + str(knn) + " shrink: " + str(shrink))
        self.recommender.fit(self.helper.URM_train, knn=knn, shrink=shrink)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def random_pop(self):
        weights = []
        i = 0
        while i < self.pop_size:
            w1 = random.randint(100, 600)
            w2 = random.randint(0, 300)
            line = [w1, w2]
            # weights.append(np.array([line]))

            if self.my_index(weights, line) == -1:
                weights.append(np.array(line))
                i += 1

        return weights

    def evaluate_pop(self):
        appo = []
        for chromosome in self.pop:
            # res = self.evaluate_chromosome(chromosome[0])
            res = self.evaluate_chromosome(chromosome)
            appo.append(res)
        return appo

    def evaluate_chromosome(self, chromosome):
        return self.step_User_CB(knn=chromosome[0], shrink=chromosome[1])

    def my_index(self, l, item):
        for i in range(len(l)):
            if (item == l[i]).all():
                return i
        return -1

    def select_parents(self):
        sorted_pop_score = sorted(self.pop_scores, reverse=False)
        probs=[]
        l = (self.pop_size*(self.pop_size+1))/2

        for i in self.pop:
            print("self.pop")
            print(self.pop)
            print("i")
            print(i)
            print("score")
            print(self.pop_scores)
            print("sorted_pop_score")
            print(sorted_pop_score)
            pos_of_i_in_pop = self.my_index(self.pop, i)
            print("pos_of_i_in_pop")
            print(pos_of_i_in_pop)
            score_of_pos = self.pop_scores[pos_of_i_in_pop]
            print("score_of_pos")
            print(score_of_pos)
            ranking = self.my_index(sorted_pop_score, score_of_pos) + 1
            print("ranking")
            print(ranking)
            prob = ranking/l
            print("prob")
            print(prob)
            probs.append(prob)
        print("------------")
        print("probs")
        print(probs)
        print("self.pop")
        print(self.pop)
        print("self.pop_scores")
        print(self.pop_scores)
        print("l")
        print(l)

        parents = [self.pop[i] for i in np.random.choice(len(self.pop), 2, p=probs)]

        return parents

    ''' 
    def select_parents(self):
        sorted_pop_score = sorted(self.pop_scores)
        probs = []
        l = (len(sorted_pop_score) * (len(sorted_pop_score) + 1)) / 2
        probs = [self.my_index(sorted_pop_score, self.pop_scores[self.my_index(self.pop, i)]) + 1 / l for i in self.pop]

        parents = [self.pop[i] for i in np.random.choice(len(self.pop), 2, p=probs)]

        return parents
    '''

    def generate_offspring(self, p1, p2):
        size = len(p1)
        offspring = np.empty((size), dtype='object')

        offspring[0] = p1[0]
        offspring[1] = p2[1]

        if self.my_index(self.new_pop, offspring) == -1:
            return offspring

        return p1

    def crossover(self, parents):
        offspring1 = self.generate_offspring(parents[0], parents[1])
        offspring2 = self.generate_offspring(parents[1], parents[0])
        # randomly mutate offsprings
        offspring1 = self.mutation(offspring1)
        offspring2 = self.mutation(offspring2)

        return offspring1, offspring2

    def mutation(self, offspring):
        if np.random.choice([True, False], 1, p=[self.p_mutation, 1-self.p_mutation]) == True:
            if self.my_index(self.new_pop, offspring) == -1:
                offspring += random.randint(0,100)
                return offspring
        return offspring


    def elitism(self):
        els = self.pop[:]
        score_c = self.pop_scores[:]

        for _ in range(4):
            index = np.argmax(score_c)
            score_c.pop(index)
            self.new_pop.append(els.pop(index))

    def run(self, max=1000, pop_size=10, p_mutation=0.1):
        self.pop_size = pop_size
        self.p_mutation = p_mutation

        self.pop = self.random_pop()
        self.pop_scores = self.evaluate_pop()

        for i in range(max):
            self.new_pop = []
            self.elitism()

            while len(self.new_pop) < len(self.pop):
                parents = self.select_parents()
                off1, off2 = self.crossover(parents)

                self.new_pop.append(off1)
                self.new_pop.append(off2)
            self.pop = self.new_pop
            self.pop_scores = self.evaluate_pop()
            print("----------------------------------------")
            print("-----------------ENDED------------------")
            print(self.pop)
            print(np.argmax(self.pop_scores))
            print("----------------------------------------")
            print("----------------------------------------")


if __name__ == "__main__":
    recommender = ItemKNNCFRecommender()
    Tuner_Singles(recommender, "ItemCF").run()