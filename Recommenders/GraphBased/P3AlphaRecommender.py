import numpy as np
from sklearn.preprocessing import normalize
from Base.Recommender_utils import check_matrix, similarityMatrixTopK
import time
import sys
import scipy.sparse as sps

RECOMMENDER_NAME = "P3alphaRecommender"


class P3AlphaRecommender(object):

    #######################################################################################
    #                                       INIT CLASS                                    #
    #######################################################################################

    def __init__(self):
        self.verbose = True

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False

    def __str__(self):
        return "P3alpha(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                            self.min_rating, self.topK, self.implicit,
                                                                            self.normalize_similarity)

    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(RECOMMENDER_NAME, string))

    def _check_format(self):

        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:

            if self.W_sparse.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("W_sparse", "csr"))

            self._W_sparse_format_checked = True

    #######################################################################################
    #                                     FIT RECOMMENDER                                 #
    #######################################################################################

    def fit(self, URM_train, topK=500, alpha=1., min_rating=0, implicit=False, normalize_similarity=False, tuning=False):

        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()
        self.n_users, self.n_items = self.URM_train.shape


        self.topK = topK
        self.alpha = alpha
        self.min_rating = min_rating
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        if self.min_rating > 0:
            self.URM_train.data[self.URM_train.data < self.min_rating] = 0
            self.URM_train.eliminate_zeros()
            if self.implicit:
                self.URM_train.data = np.ones(self.URM_train.data.size, dtype=np.float32)

        # Pui is the row-normalized urm
        Pui = normalize(self.URM_train, norm='l1', axis=1)

        # Piu is the column-normalized, "boolean" urm transposed
        X_bool = self.URM_train.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)
        # ATTENTION: axis is still 1 because i transposed before the normalization
        Piu = normalize(X_bool, norm='l1', axis=1)
        del (X_bool)

        # Alfa power
        if self.alpha != 1.:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Final matrix is computed as Pui * Piu * Pui
        # Multiplication unpacked for memory usage reasons
        block_dim = 200
        d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        for current_block_start_row in range(0, Pui.shape[1], block_dim):

            if current_block_start_row + block_dim > Pui.shape[1]:
                block_dim = Pui.shape[1] - current_block_start_row

            similarity_block = d_t[current_block_start_row:current_block_start_row + block_dim, :] * Pui
            similarity_block = similarity_block.toarray()

            for row_in_block in range(block_dim):
                row_data = similarity_block[row_in_block, :]
                row_data[current_block_start_row + row_in_block] = 0

                best = row_data.argsort()[::-1][:self.topK]

                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))

                    rows[numCells] = current_block_start_row + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

            if time.time() - start_time_printBatch > 60:
                self._print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Rows per second: {:.0f}".format(
                    current_block_start_row,
                    100.0 * float(current_block_start_row) / Pui.shape[1],
                    (time.time() - start_time) / 60,
                    float(current_block_start_row) / (time.time() - start_time)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(Pui.shape[1], Pui.shape[1]))

        if self.normalize_similarity:
            self.W_sparse = normalize(self.W_sparse, norm='l1', axis=1)

        if self.topK != False:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK)

        self.W_sparse = check_matrix(self.W_sparse, format='csr')
        self.similarityProduct = self.URM_train.dot(self.W_sparse)

    #######################################################################################
    #                                    RUN RECOMMENATION                                #
    #######################################################################################

    def get_expected_ratings(self, user_id):
        expected_ratings = self.similarityProduct[user_id].toarray().ravel()
        return np.squeeze(np.asarray(expected_ratings))

    def recommend(self, user_id, at=10):
        # compute the scores using the dot product
        expected_ratings = self.get_expected_ratings(user_id)
        ranking = expected_ratings.argsort()[::-1]
        unseen_items_mask = np.in1d(ranking, self.URM_train[user_id].indices, assume_unique=True, invert=True)
        ranking = ranking[unseen_items_mask]
        return ranking[:at]