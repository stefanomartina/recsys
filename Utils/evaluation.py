import numpy as np
import scipy.sparse as sps


# relevant items: items in the test set
# precision: how many of the recommended items are relevant
from tqdm import tqdm


def precision(is_relevant, relevant_items):

    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


# recall: how many of the relevant items I was able to recommend
def recall(is_relevant, relevant_items):

    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(is_relevant, relevant_items):

    # is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(urm_test, recommender_object, at=10):

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_map = 0.0

    num_eval = 0

    urm_test = sps.csr_matrix(urm_test)

    n_users = urm_test.shape[0]

    for user_id in tqdm(range(n_users)):

        start_pos = urm_test.indptr[user_id]
        end_pos = urm_test.indptr[user_id + 1]

        if end_pos-start_pos > 0:

            relevant_items = urm_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_map += MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_map /= num_eval

    print("Recommender performance is: Precision = {:.6f}, Recall = {:.6f}, MAP = {:.6f}".format(
        cumulative_precision, cumulative_recall, cumulative_map))

    return cumulative_map

def evaluate_algorithm_classes(urm_test, user_to_recommend, recommender_object, at=10):

    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_map = 0.0

    num_eval = 0

    urm_test = sps.csr_matrix(urm_test)

    n_users = user_to_recommend

    for user_id in tqdm(n_users):

        start_pos = urm_test.indptr[user_id]
        end_pos = urm_test.indptr[user_id + 1]

        if end_pos-start_pos > 0:

            relevant_items = urm_test.indices[start_pos:end_pos]

            recommended_items = recommender_object.recommend(user_id, at=at)
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_precision += precision(is_relevant, relevant_items)
            cumulative_recall += recall(is_relevant, relevant_items)
            cumulative_map += MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_map /= num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision, cumulative_recall, cumulative_map))

    return cumulative_map
