import os
import sys
import time
import warnings
import numpy as np
import scipy.sparse as sps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from Base.BaseFunction import BaseFunction
from Base.Recommender_utils import check_matrix
from Recommenders.BaseRecommender import BaseRecommender
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

RECOMMENDER_NAME = "SLIMElasticNetRecommender"
SIMILARITY_PATH = "/SimilarityProduct/SlimElastic_similarity.npz"

class SLIMElasticNetRecommender(BaseRecommender):

    def run_fit(self):
        # Display ConvergenceWarning only once and not for every item it occurs
        warnings.simplefilter("once", category=ConvergenceWarning)

        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1e-4,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        URM_train = check_matrix(self.URM, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            if y.sum() == 0.0:
                continue

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if time.time() - start_time_printBatch > 300 or currentItem == n_items - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} {}. Items per second: {:.2f}".format(
                    currentItem + 1,
                    100.0 * float(currentItem + 1) / n_items,
                    new_time_value,
                    new_time_unit,
                    float(currentItem) / elapsed_time))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(n_items, n_items), dtype=np.float32)

    def fit(self, URM, verbose=True, l1_ratio=1.0, alpha = 1.0, positive_only=True, topK = 494, tuning=False, similarity_path=SIMILARITY_PATH):

        self.URM = URM
        self.l1_ratio = l1_ratio
        self.positive_only = positive_only
        self.topK = topK
        #1e-4
        self.helper = BaseFunction()

        if tuning:
            if not os.path.exists(os.getcwd() + similarity_path):
                self.run_fit()
                self.helper.export_similarity_matrix(os.getcwd() + similarity_path, self.W_sparse,
                                                     name=RECOMMENDER_NAME)
            self.W_sparse = self.helper.import_similarity_matrix(os.getcwd() + similarity_path)
            self.similarityProduct = self.URM.dot(self.W_sparse)

        else:
            self.run_fit()
            self.similarityProduct = self.URM.dot(self.W_sparse)