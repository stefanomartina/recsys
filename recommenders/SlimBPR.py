import time
import numpy as np
from utils import similarityMatrixTopK as sim


class SlimBPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def sampleTriplet(self):
        user_id = np.random.choice(self.eligibleUsers)

        userSeenItems = self.URM_mask[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)
        neg_item_id = 0
        negItemSelected = False

        while not negItemSelected:
            neg_item_id = np.random.randint(0, self.n_items)

            if neg_item_id not in userSeenItems:
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):
        learning_rate = 1e-3

        numPositiveIteractions = int(self.URM_mask.nnz * 0.01)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        for num_sample in range(numPositiveIteractions):
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.URM_mask[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update
            self.similarity_matrix[positive_item_id, userSeenItems] += learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, userSeenItems] -= learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self, URM, learning_rate=0.01, epochs=75):
        self.URM = URM
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.URM_mask = self.URM.copy()
        #  self.URM_mask.data[self.URM_mask.data <= 3] = 0
        self.URM_mask.eliminate_zeros()

        self.n_users = self.URM_mask.shape[0]
        self.n_items = self.URM_mask.shape[1]

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

        # Extract users having at least one interaction to choose from
        self.eligibleUsers = []

        for user_id in range(self.n_users):

            start_pos = self.URM_mask.indptr[user_id]
            end_pos = self.URM_mask.indptr[user_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

        for numEpoch in range(self.epochs):
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T
        self.similarity_matrix = sim.similarityMatrixTopK(self.similarity_matrix, k=100)

    def recommend(self, user_id, at=10, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

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