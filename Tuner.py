import numpy as np

from Recommenders import HybridItemCF_ItemCB
from Recommenders.Slim.Cython import SLIM_BPR_Cython
from Runner import Runner
from Utils import evaluation


class Tuner(Runner):

    def __init__(self, recommender, name):
        super(Tuner, self).__init__(self, recommender, name)
        self.recommender = recommender


    def step_HybridItemCF_ItemCB(self, topK, list_ICM, shrink, similarity):
        print("----------------------------------------")
        print("topk: " + str(topK) + " shrink: " + str(shrink) + " similarity: " + similarity)
        self.recommender.fit(self.URM_train, list_ICM, topK=topK, shrink=shrink, similarity=similarity, normalize=True)
        evaluation.evaluate_algorithm(self.URM_test, self.recommender, at=10)
        print("----------------------------------------")

    def step_SlimBPR_Cython(self, topK, learning_rate, L1, L2, sgd_mode, gamma=0.995, beta1=0.9, beta2=0.999):
        print("----------------------------------------")
        print("topk: " + str(topK) + " learning_rate: " + str(learning_rate) + " L1: " + str(L1) + " L2: " + str(L2) + " beta1: " + str(beta1) + " beta2: " + str(beta2) +
             " gamma: " + str(gamma) + " sgd_mode: " + sgd_mode)

        self.recommender.fit(self.URM_train, topK=topK, learning_rate=learning_rate, lambda_i = L1, lambda_j = L2, beta_1 = beta1, beta_2 = beta2, gamma = gamma, sgd_mode = sgd_mode)
        evaluation.evaluate_algorithm(self.URM_test, self.recommender, at=10)
        print("----------------------------------------")


    def run(self, requires_icm=False, requires_ucm=False):
        self.get_URM()
        self.split_dataset_loo()
        self.get_target_users()

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


if __name__ == "__main__":
    recommender = SLIM_BPR_Cython.SLIM_BPR_Cython()
    Tuner(recommender, "SlimBPR_Cython").run()
