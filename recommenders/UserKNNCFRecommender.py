""" Created on 23/10/17     @author: Maurizio Ferrari Dacrema """
from Base.DataIO import DataIO
from Base.Recommender_utils import check_matrix
from Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
from Base.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCFRecommender(object):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCFRecommender"
    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

# DEFINITION of RECOMMENDER

    def _check_format(self):

        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                self._print(
                    "PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format(
                        "URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:

            if self.W_sparse.getformat() != "csr":
                self._print(
                    "PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format(
                        "W_sparse", "csr"))

            self._W_sparse_format_checked = True

    def _get_cold_user_mask(self):
        return self._cold_user_mask

    def _get_cold_item_mask(self):
        return self._cold_item_mask

    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.RECOMMENDER_NAME, string))

    def fit(self, URM_train, verbose=True, topK=50, shrink=100, similarity='cosine', normalize=True,
            feature_weighting="none", **similarity_args):

        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape
        self.verbose = verbose

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold users.".format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum() / self.n_users * 100))

        self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0

        if self._cold_item_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold items.".format(
                self._cold_item_mask.sum(), self._cold_item_mask.sum() / self.n_items * 100))

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train.T).T
            self.URM_train = check_matrix(self.URM_train, 'csr')

        similarity = Compute_Similarity(self.URM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):
        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)


 # COMPUTE and FILTER RECOMMENDATION LIST
    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch

    def _remove_custom_items_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch

    def _remove_seen_on_scores(self, user_id, scores):
        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"
        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
        scores[seen] = -np.inf
        return scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        :param user_id_array:       array containing the user indices whose recommendations need to be computed
        :param items_to_compute:    array containing the items whose scores are to be computed.
                                            If None, all items are computed, otherwise discarded items will have as score -np.inf
        :return:                    array (len(user_id_array), n_items) with the score.
        """
        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32) * np.inf
            item_scores_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train).toarray()

        return item_scores

    def recommend(self, user_id_array, at =10, cutoff=None, remove_seen_flag=True, items_to_compute=None,
              remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
    # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list[:at]



# LOAD and SAVE
    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"W_sparse": self.W_sparse}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
             self.__setattr__(attrib_name, data_dict[attrib_name])

        self._print("Loading complete")
