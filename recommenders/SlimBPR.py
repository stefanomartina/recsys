import numpy as np
import time
import scipy.sparse as sps
from recommenders import TopPopRecommender

class SLIM_BPR_Recommender(object):

    def check_matrix(self, X, format='csc', dtype=np.float32):
        if format == 'csc' and not isinstance(X, sps.csc_matrix):
            return X.tocsc().astype(dtype)
        elif format == 'csr' and not isinstance(X, sps.csr_matrix):
            return X.tocsr().astype(dtype)
        elif format == 'coo' and not isinstance(X, sps.coo_matrix):
            return X.tocoo().astype(dtype)
        elif format == 'dok' and not isinstance(X, sps.dok_matrix):
            return X.todok().astype(dtype)
        elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
            return X.tobsr().astype(dtype)
        elif format == 'dia' and not isinstance(X, sps.dia_matrix):
            return X.todia().astype(dtype)
        elif format == 'lil' and not isinstance(X, sps.lil_matrix):
            return X.tolil().astype(dtype)

        elif format == 'npy':
            if sps.issparse(X):
                return X.toarray().astype(dtype)
            else:
                return np.array(X)

        elif isinstance(X, np.ndarray):
            X = sps.csr_matrix(X, dtype=dtype)
            X.eliminate_zeros()
            return self.check_matrix(X, format=format, dtype=dtype)
        else:
            return X.astype(dtype)

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligibleUsers)

        # Get user seen items and choose one
        userSeenItems = self.URM_mask[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def similarityMatrixTopK(self, item_weights, forceSparseOutput=True, k=100, verbose=False, inplace=True):

        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        start_time = time.time()

        if verbose:
            print("Generating topK matrix")

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        if not sparse_weights:

            idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

            if inplace:
                W = item_weights
            else:
                W = item_weights.copy()

            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            W[not_top_k, np.arange(nitems)] = 0.0

            if forceSparseOutput:
                W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

                if verbose:
                    print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

                return W_sparse

            if verbose:
                print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W

        else:
            # iterate over each column and keep only the top-k similar items
            data, rows_indices, cols_indptr = [], [], []

            item_weights = self.check_matrix(item_weights, format='csc', dtype=np.float32)

            for item_idx in range(nitems):
                cols_indptr.append(len(data))

                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx + 1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

                non_zero_data = column_data != 0

                idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
                top_k_idx = idx_sorted[-k:]

                data.extend(column_data[non_zero_data][top_k_idx])
                rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

            cols_indptr.append(len(data))

            # During testing CSR is faster
            W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
            W_sparse = W_sparse.tocsr()

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz * 0.01)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.URM_mask[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update
            self.similarity_matrix[positive_item_id, userSeenItems] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, userSeenItems] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if (time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self, URM, learning_rate=0.01, epochs=300):
        self.URM = URM
        self.URM_mask = self.URM.copy()
        self.n_users = self.URM_mask.shape[0]
        self.n_items = self.URM_mask.shape[1]
        self.TP = TopPopRecommender.TopPopRecommender()
        self.TP.fit(self.URM)
        self.similarity_matrix = np.zeros((self.n_items, self.n_items))
        self.count_cold = 0


        self.URM_mask.eliminate_zeros()

        self.eligibleUsers = []

        for user_id in range(self.n_users):

            start_pos = self.URM_mask.indptr[user_id]
            end_pos = self.URM_mask.indptr[user_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

        self.learning_rate = learning_rate
        self.epochs = epochs

        for numEpoch in range(self.epochs):
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = self.similarityMatrixTopK(self.similarity_matrix, k=10)

    def recommend(self, user_id, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()
        summed_score = scores.sum(axis=0)

        # "Hybrid version with TOP-POP"
        if summed_score == 0:
            self.count_cold += 1
            return self.TP.recommend(user_id)

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores