#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Created on 22/11/17     @author: Maurizio Ferrari Dacrema """

from Recommenders.Collaborative import ItemKNNCFRecommender
from Recommenders.ContentBased import UserCBFKNNRecommender, ItemCBFKNNRecommender
from Recommenders.NonPersonalizedRecommender import TopPopRecommender, RandomRecommender
from Recommenders.Slim.SlimBPR.Cython import SLIM_BPR_Cython
# from Recommenders.MatrixFactorizationRecommenders.Cython.MatrixFactorization_Cython import \
#     MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
# from Recommenders.MatrixFactorizationRecommenders.PureSVD import PureSVDRecommender

import os, multiprocessing
from functools import partial

from Base.BaseFunction import BaseFunction
from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    dataReader = BaseFunction()
    dataReader.get_URM()
    dataReader.split_dataset_loo()

    output_folder_path = "result_experiments/"
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # URM_train = dataReader.get_URM_train()
    # URM_validation = dataReader.get_URM_validation()
    # URM_test = dataReader.get_URM_test()

    collaborative_algorithm_list = [
        # RandomRecommender,
        TopPopRecommender,
        ItemKNNCFRecommender,
        UserCBFKNNRecommender,
        ItemCBFKNNRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        SLIM_BPR_Cython,
        # SLIMElasticNetRecommender
    ]

    from Base.Evaluation.Evaluator import EvaluatorHoldout

    evaluator_validation = EvaluatorHoldout(dataReader.URM_test, cutoff_list=[5])
    evaluator_test = EvaluatorHoldout(dataReader.URM_train, cutoff_list=[5, 10])

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=dataReader.URM_all,
                                                       metric_to_optimize="MAP",
                                                       n_cases=10,
                                                       evaluator_validation_earlystopping=evaluator_validation,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=evaluator_test,
                                                       output_folder_path=output_folder_path,
                                                       similarity_type_list=["cosine"],
                                                       parallelizeKNN=False)

    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)


    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #


if __name__ == '__main__':
    read_data_split_and_search()
