from datetime import time
from CythonCompiler.run_compile_subprocess import run_compile_subprocess
import numpy as np
import time
import sys
from Base.BaseTempFolder import BaseTempFolder
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit


RECOMMENDER_NAME = "AbstractMatrixRecommender"

class BaseMatrixFactorization():

    def __init__(self, verbose=True, algorithm_name = None):

        self.verbose = verbose
        self.normalize = False
        self.algorithm_name = algorithm_name

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        self.use_bias = False

    def get_early_stopping_final_epochs_dict(self):
        return {"epochs": self.epochs_best}

    def _train_with_early_stopping(self, epochs_max, epochs_min = 0,
                                   validation_every_n=None, stop_on_validation=False,
                                   validation_metric=None, lower_validations_allowed=None, evaluator_object=None,
                                   algorithm_name="Incremental_Training_Early_Stopping"):

        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0

        epochs_current = 0

        while epochs_current < epochs_max and not convergence:

            self._run_epoch(epochs_current)

            # If no validation required, always keep the latest
            if evaluator_object is None:

                self.epochs_best = epochs_current

            # Determine whether a validaton step is required
            elif (epochs_current + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._prepare_model_for_validation()

                # If the evaluator validation has multiple cutoffs, choose the first one
                results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                results_run = results_run[list(results_run.keys())[0]]

                print("{}: {}".format(algorithm_name, results_run_string))

                # Update optimal model
                current_metric_value = results_run[validation_metric]

                if not np.isfinite(current_metric_value):
                    if isinstance(self, BaseTempFolder):
                        # If the recommender uses BaseTempFolder, clean the temp folder
                        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

                    assert False, "{}: metric value is not a finite number, terminating!".format(RECOMMENDER_NAME)


                if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                    print("{}: New best model found! Updating.".format(algorithm_name))

                    self.best_validation_metric = current_metric_value

                    self._update_best_model()

                    self.epochs_best = epochs_current +1
                    lower_validatons_count = 0

                else:
                    lower_validatons_count += 1


                if stop_on_validation and lower_validatons_count >= lower_validations_allowed and epochs_current >= epochs_min:
                    convergence = True

                    elapsed_time = time.time() - start_time
                    new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                    print("{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                        algorithm_name, epochs_current+1, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))


            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
                algorithm_name, epochs_current+1, epochs_max, new_time_value, new_time_unit))

            epochs_current += 1

            sys.stdout.flush()
            sys.stderr.flush()

        # If no validation required, keep the latest
        if evaluator_object is None:

            self._prepare_model_for_validation()
            self._update_best_model()


        # Stop when max epochs reached and not early-stopping
        if not convergence:
            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if evaluator_object is not None and self.best_validation_metric is not None:
                print("{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current, validation_metric, self.epochs_best, self.best_validation_metric, new_time_value, new_time_unit))
            else:
                print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current, new_time_value, new_time_unit))

    def _prepare_model_for_validation(self):
        self.USER_factors = self.cythonEpoch.get_USER_factors()
        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _run_epoch(self, num_epoch):
       self.cythonEpoch.epochIteration_Cython()

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        file_subfolder = "/Recommenders/MatrixFactorization/Cython"
        file_to_compile_list = ['MatrixFactorization_Cython_Epoch.pyx']

        run_compile_subprocess(file_subfolder, file_to_compile_list)

        print("{}: Compiled module {} in subfolder: {}".format(RECOMMENDER_NAME, file_to_compile_list, file_subfolder))

        # Command to run compilation script
        # python compile_script.py MatrixFactorization_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a MatrixFactorization_Cython_Epoch.pyx

    def fit(self, URM_train, recompile_cython=False, algorithm_name="MF_BPR", epochs=100, batch_size=1000,
            num_factors=30, positive_threshold_BPR=None, learning_rate=0.002, use_bias=True, sgd_mode='adagrad',
            negative_interactions_quota=0.0, init_mean=0.0, init_std_dev=0.1,
            user_reg=0.71, item_reg=0.2, bias_reg=0.5, positive_reg=0.0, negative_reg=0.0, random_seed=None,
            **earlystopping_kwargs):

        self.URM = URM_train
        self.n_users, self.n_items = self.URM.shape
        self.num_factors = num_factors
        self.use_bias = use_bias
        self.sgd_mode = sgd_mode
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate

        assert negative_interactions_quota >= 0.0 and negative_interactions_quota < 1.0, \
            "{}: negative_interactions_quota must be a float value >=0 and < 1.0, provided was '{}'".format(
                RECOMMENDER_NAME, negative_interactions_quota)
        self.negative_interactions_quota = negative_interactions_quota

        # Import compiled module
        from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython_Epoch import MatrixFactorization_Cython_Epoch

        if self.algorithm_name in ["FUNK_SVD", "ASY_SVD"]:

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                item_reg=item_reg,
                                                                bias_reg=bias_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                init_mean=init_mean,
                                                                negative_interactions_quota=negative_interactions_quota,
                                                                init_std_dev=init_std_dev,
                                                                verbose=self.verbose,
                                                                random_seed=random_seed)

        elif self.algorithm_name == "MF_BPR":

            # Select only positive interactions
            URM_train_positive = self.URM.copy()

            if self.positive_threshold_BPR is not None:
                URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
                URM_train_positive.eliminate_zeros()

                assert URM_train_positive.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(URM_train_positive,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                positive_reg=positive_reg,
                                                                negative_reg=negative_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                init_mean=init_mean,
                                                                init_std_dev=init_std_dev,
                                                                verbose=self.verbose,
                                                                random_seed=random_seed)
        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.algorithm_name,
                                        **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        if self.use_bias:
            self.USER_bias = self.USER_bias_best
            self.ITEM_bias = self.ITEM_bias_best
            self.GLOBAL_bias = self.GLOBAL_bias_best

        self.matrix_scores = np.dot(self.USER_factors, self.ITEM_factors.T)
        sys.stdout.flush()


    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch

    def _remove_custom_items_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM.indices[self.URM.indptr[user_id]:self.URM.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def get_expected_ratings(self, user_id):
        expected_scores = (self.matrix_scores[user_id]).ravel()
        return expected_scores

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]
        scores[user_profile] -= np.inf

        return scores

    def recommend(self, user_id, at=10, exclude_seen=True):
        expected_scores = self.get_expected_ratings(user_id)

        if exclude_seen:
            expected_scores = self.filter_seen(user_id, expected_scores)
        ranking = expected_scores.argsort()[::-1]

        return ranking[:at]




class MatrixFactorization_BPR_Cython(BaseMatrixFactorization):

    RECOMMENDER_NAME = "MatrixFactorization_BPR_Cython_Recommender"

    def __init__(self):
        super(MatrixFactorization_BPR_Cython, self).__init__(algorithm_name="MF_BPR")

    def fit(self, URM_train, recompile_cython=False, algorithm_name="MF_BPR", epochs=600, batch_size=1000,
            num_factors=30, positive_threshold_BPR=None, learning_rate=0.002, use_bias=True, sgd_mode='adagrad',
            negative_interactions_quota=0.0, init_mean=0.0, init_std_dev=0.1,
            user_reg=0.71, item_reg=0.2, bias_reg=0.5, positive_reg=0.0, negative_reg=0.0, random_seed=None,
            **earlystopping_kwargs):

        super(MatrixFactorization_BPR_Cython, self).fit(URM_train, use_bias=False, negative_interactions_quota=0.0)

class MatrixFactorization_FunkSVD_Cython(BaseMatrixFactorization):
    RECOMMENDER_NAME = "MatrixFactorization_FunkSVD_Cython_Recommender"

    def __init__(self):
        super(MatrixFactorization_FunkSVD_Cython, self).__init__(algorithm_name="FUNK_SVD")

    def fit(self, URM_train, recompile_cython=False, algorithm_name="MF_BPR", epochs=600, batch_size=1000,
            num_factors=30, positive_threshold_BPR=None, learning_rate=0.002, use_bias=True, sgd_mode='adagrad',
            negative_interactions_quota=0.0, init_mean=0.0, init_std_dev=0.1,
            user_reg=0.71, item_reg=0.2, bias_reg=0.5, positive_reg=0.0, negative_reg=0.0, random_seed=None,
            **earlystopping_kwargs):

        super(MatrixFactorization_FunkSVD_Cython, self).fit(URM_train)

class MatrixFactorization_AsySVD_Cython(BaseMatrixFactorization):
    RECOMMENDER_NAME = "MatrixFactorization_AsySVD_Cython_Recommender"

    def __init__(self):
        super(MatrixFactorization_AsySVD_Cython, self).__init__(algorithm_name="ASY_SVD")

    def fit(self, URM_train, recompile_cython=False, algorithm_name="ASY_SVD", epochs=600, batch_size=1000,
            num_factors=30, positive_threshold_BPR=None, learning_rate=0.002, use_bias=True, sgd_mode='adagrad',
            negative_interactions_quota=0.0, init_mean=0.0, init_std_dev=0.1,
            user_reg=0.71, item_reg=0.2, bias_reg=0.5, positive_reg=0.0, negative_reg=0.0, random_seed=None,
            **earlystopping_kwargs):

        super(MatrixFactorization_AsySVD_Cython, self).fit(URM_train, batch_size=1, algorithm_name="ASY_SVD")

    def _prepare_model_for_validation(self):
        """
        AsymmetricSVD Computes two |n_items| x |n_features| matrices of latent factors
        ITEM_factors_Y must be used to estimate user's latent factors via the items they interacted with

        :return:
        """

        self.ITEM_factors_Y = self.cythonEpoch.get_USER_factors()
        self.USER_factors = self._estimate_user_factors(self.ITEM_factors_Y)

        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.ITEM_factors_Y_best = self.ITEM_factors_Y.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _estimate_user_factors(self, ITEM_factors_Y):

        profile_length = np.ediff1d(self.URM.indptr)
        profile_length_sqrt = np.sqrt(profile_length)

        # Estimating the USER_factors using ITEM_factors_Y
        if self.verbose:
            print("{}: Estimating user factors... ".format(self.algorithm_name))

        USER_factors = self.URM.dot(ITEM_factors_Y)

        #Divide every row for the sqrt of the profile length
        for user_index in range(self.n_users):

            if profile_length_sqrt[user_index] > 0:

                USER_factors[user_index, :] /= profile_length_sqrt[user_index]

        if self.verbose:
            print("{}: Estimating user factors... done!".format(self.algorithm_name))

        return USER_factors

