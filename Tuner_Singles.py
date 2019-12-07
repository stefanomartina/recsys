import random
import numpy as np
import time

from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Collaborative.UserKNNCFRecommender import UserKNNCFRecommender
from Hybrid import HybridRecommender

from Base.BaseFunction import BaseFunction
from Utils import evaluation


class Tuner_Singles():

    def __init__(self, recommender, name):
        self.recommender = recommender
        self.name = name
        self.helper = BaseFunction()
        self.helper.get_URM()
        self.helper.get_ICM()
        self.helper.get_UCM()
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

    def step_weight(self, w1, w2):
        start_time = time.time()
        print("----------------------------------------")
        print("HybridCombination: " + self.name)
        print([w1, w2])
        print("----------------------------------------")
        list_UCM = [self.helper.UCM_age, self.helper.UCM_region]
        list_ICM = [self.helper.ICM, self.helper.ICM_price, self.helper.ICM_asset]
        self.recommender.fit(self.helper.URM_train, [w1, w2], list_ICM=list_ICM, list_UCM=list_UCM,  tuning=False)
        cumulative = evaluation.evaluate_algorithm(self.helper.URM_test, self.recommender, at=10)
        elapsed_time = time.time() - start_time
        print("----------------" + str(elapsed_time) + "----------------")
        return cumulative

    def random_pop(self):
        weights = []
        one = 1
        """"
        for i in range(self.pop_size):
            w1 = random.randint(100, 600)
            w2 = random.randint(0, 300)
            line = [w1, w2]
            weights.append(np.array(line))
        """
        for i in range(self.pop_size):
            w1 = random.uniform(0, 0.9)
            w2 = one - w1
            line = [w1, w2]
            weights.append(np.array(line))

        return weights

    def evaluate_pop(self):
        appo = []
        for chromosome in self.pop:
            res = self.evaluate_chromosome(chromosome)
            appo.append(res)
        return appo

    def evaluate_chromosome(self, chromosome):
        return self.step_weight(w1=chromosome[0], w2=chromosome[1])

    def my_index(self, l, item):
        for i in range(len(l)):
            if (item == l[i]).all():
                return i
        return -1

    def select_parents(self):
        sorted_pop_score = sorted(self.pop_scores, reverse=False)
        probs=[]
        taken_pop = [False]*self.pop_size
        taken_score = [False]*self.pop_size

        l = (self.pop_size*(self.pop_size+1))/2

        for i in self.pop:
            pos_of_i_in_pop = self.my_index(self.pop, i)
            while taken_pop[pos_of_i_in_pop]:
                pos_of_i_in_pop += self.my_index(self.pop[pos_of_i_in_pop + 1 : ], i) + 1

            score_of_pos = self.pop_scores[pos_of_i_in_pop]
            ranking = self.my_index(sorted_pop_score, score_of_pos)

            while taken_score[ranking]:
                ranking += self.my_index(sorted_pop_score[ranking + 1 : ], score_of_pos) +1

            taken_score[ranking] = True
            taken_pop[pos_of_i_in_pop] = True
            prob = (ranking+1)/l
            probs.append(prob)

        parents = [self.pop[i] for i in np.random.choice(len(self.pop), 2, p=probs)]

        return parents

    def generate_offspring(self, p1, p2):
        size = len(p1)
        offspring = np.empty((size), dtype='object')

        offspring[0] = p1[0]
        offspring[1] = p2[1]

        return offspring

    def crossover(self, parents):
        offspring1 = self.generate_offspring(parents[0], parents[1])
        offspring2 = self.generate_offspring(parents[1], parents[0])
        offspring1 = self.mutation(offspring1)
        offspring2 = self.mutation(offspring2)

        return offspring1, offspring2

    def mutation(self, offspring):
        if np.random.choice([True, False], 1, p=[self.p_mutation, 1-self.p_mutation]) == True:
            offspring += random.randint(0,100)
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
            print("-----------------ENDED------------------")
            print(self.pop)
            print(np.argmax(self.pop_scores))
            print("----------------------------------------")


if __name__ == "__main__":
    recommender = HybridRecommender.HybridRecommender("Combo5")
    Tuner_Singles(recommender, "Hybrid").run()