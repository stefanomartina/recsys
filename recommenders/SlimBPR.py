import time
import numpy as np
from utils import similarityMatrixTopK as sim
import sys
from scipy.special import expit

class SlimBPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def sampleUser(self):
        while (True):

            user_id = np.random.randint(0, self.n_users)
            numSeenItems = self.URM_train[user_id].nnz

            if (numSeenItems > 0 and numSeenItems < self.n_items):
                return user_id

    def sampleItemPair(self, user_id):
        userSeenItems = self.URM_train[user_id].indices

        pos_item_id = userSeenItems[np.random.randint(0, len(userSeenItems))]

        while (True):

            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                return pos_item_id, neg_item_id

    def sampleTriple(self):
        user_id = self.sampleUser()
        pos_item_id, neg_item_id = self.sampleItemPair(user_id)

        return user_id, pos_item_id, neg_item_id

    def updateFactors(self, user_id, pos_item_id, neg_item_id):

        # Calculate current predicted score
        userSeenItems = self.URM_train[user_id].indices
        prediction = 0

        for userSeenItem in userSeenItems:
            prediction += self.S[pos_item_id, userSeenItem] - self.S[neg_item_id, userSeenItem]

        x_uij = prediction
        logisticFunction = expit(-x_uij)

        # Update similarities for all items except those sampled
        for userSeenItem in userSeenItems:

            # For positive item is PLUS logistic minus lambda*S
            if (pos_item_id != userSeenItem):
                update = logisticFunction - self.lambda_i * self.S[pos_item_id, userSeenItem]
                self.S[pos_item_id, userSeenItem] += self.learning_rate * update

            # For positive item is MINUS logistic minus lambda*S
            if (neg_item_id != userSeenItem):
                update = - logisticFunction - self.lambda_j * self.S[neg_item_id, userSeenItem]
                self.S[neg_item_id, userSeenItem] += self.learning_rate * update

    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = self.URM_train.nnz
        start_time = time.time()

        # Uniform user sampling without replacement
        for numSample in range(numPositiveIteractions):

            user_id, pos_item_id, neg_item_id = self.sampleTriple()
            self.updateFactors(user_id, pos_item_id, neg_item_id)

            if (numSample % 5000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.4f} seconds".format(numSample,
                                                                          100.0 * float(
                                                                              numSample) / numPositiveIteractions,
                                                                          time.time() - start_time))

                sys.stderr.flush()

                start_time = time.time()

    def fit(self, URM, epochs=15, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05):
        self.URM_train = URM
        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate
        self.normalize = False
        self.sparse_weights = False
        self.S = np.random.random((self.n_items, self.n_items)).astype('float32')
        self.S[np.arange(self.n_items), np.arange(self.n_items)] = 0

        start_time_train = time.time()

        for currentEpoch in range(epochs):
            start_time_epoch = time.time()

            self.epochIteration()
            print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch + 1, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))

        print("Train completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))

        # The similarity matrix is learnt row-wise
        # To be used in the product URM*S must be transposed to be column-wise
        self.W = self.S.T

        del self.S

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