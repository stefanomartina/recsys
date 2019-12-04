#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Created on 07/09/17     @author: Maurizio Ferrari Dacrema """

import sys
import numpy as np
from Base.Recommender_utils import similarityMatrixTopK, check_matrix


class SLIM_BPR_Cython(object):
    RECOMMENDER_NAME = "SLIM_BPR_Recommender"

    #######################################################################################
    #                                   INIT SLIM_BPR                                     #
    #######################################################################################

    def __init__(self, positive_threshold=1, recompile_cython=False, final_model_sparse_weights=True,
                 train_with_sparse_weights=False, symmetric=True, epochs = 300,
                batch_size = 100, lambda_i = 0.01, lambda_j = 0.001, learning_rate = 0.01, topK = 200,
                sgd_mode = 'adagrad', gamma = 0.995, beta_1 = 0.9, beta_2 = 0.999):

        #### Retreiving parameters for fitting #######
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.topK = topK
        self.sgd_mode = sgd_mode
        self.gamma = gamma
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.symmetric = symmetric
        #############################################

        self.normalize = False
        self.positive_threshold = positive_threshold

        self.train_with_sparse_weights = train_with_sparse_weights
        self.sparse_weights = final_model_sparse_weights

        if self.train_with_sparse_weights:
            self.sparse_weights = True

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    #######################################################################################
    #                                     RUN FITTNG                                      #
    #######################################################################################

    def fit(self, URM_train):
        print("Fitting Slim...")
        self.__init__()
        self.URM = URM_train

        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]

        # Select only positive interactions
        URM_train_positive = self.URM.copy()
        self.URM_mask = self.URM.copy()

        self.URM_mask.data = self.URM_mask.data >= self.positive_threshold
        self.URM_mask.eliminate_zeros()

        assert self.URM_mask.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

        # Start fitting
        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()

        from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch
        self.cythonEpoch = SLIM_BPR_Cython_Epoch(self.URM_mask,
                                                 train_with_sparse_weights=self.train_with_sparse_weights,
                                                 final_model_sparse_weights=self.sparse_weights,
                                                 topK=self.topK,
                                                 learning_rate=self.learning_rate,
                                                 li_reg=self.lambda_i,
                                                 lj_reg=self.lambda_j,
                                                 batch_size=self.batch_size,
                                                 symmetric=self.symmetric,
                                                 sgd_mode=self.sgd_mode,
                                                 gamma=self.gamma,
                                                 beta_1=self.beta_1,
                                                 beta_2=self.beta_2)

        self._initialize_incremental_model()
        self.epochs_best = 0
        currentEpoch = 0

        while currentEpoch < self.epochs:
            self._run_epoch()
            self._update_best_model()
            currentEpoch += 1

        self.get_S_incremental_and_set_W()
        self.cythonEpoch._dealloc()
        sys.stdout.flush()

        self.score = self.URM.dot(self.W_sparse)

    def _initialize_incremental_model(self):
        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()

    def _update_incremental_model(self):
        self.get_S_incremental_and_set_W()

    def _update_best_model(self):
        self.S_best = self.S_incremental.copy()

    def _run_epoch(self):
        self.cythonEpoch.epochIteration_Cython()

    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental

        else:
            self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.topK)

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        file_subfolder = "/Slim/Cython"
        file_to_compile_list = ['SLIM_BPR_Cython_Epoch.pyx']

        run_compile_subprocess(file_subfolder, file_to_compile_list)

        print("{}: Compiled module {} in subfolder: {}".format(self.RECOMMENDER_NAME, file_to_compile_list,
                                                               file_subfolder))

        # Command to run compilation script
        # python compile_script.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a SLIM_BPR_Cython_Epoch.pyx

    def get_expected_ratings(self, user_id):
        expected_ratings = self.score[user_id].todense()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id, at=10):

        # compute the scores using the dot product
        scores = self.get_expected_ratings(user_id)
        ranking = scores.argsort()[::-1]
        unseen_items_mask = np.in1d(ranking, self.URM[user_id].indices, assume_unique=True, invert=True)
        ranking = ranking[unseen_items_mask]
        return ranking[:at]
