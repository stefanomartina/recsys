""" @author: Simone Lanzillotta, Stefano Martina """

import pandas as pd
from Base.Similarity.Cython.Compute_Similarity_Cython import Compute_Similarity_Cython
import scipy.sparse as sps
import numpy as np
from sklearn import preprocessing, datasets
import os

class BaseFunction:

    def __init__(self):
        self.userlist_unique = []

        # URM --------------------------
        self.userlist_urm = None
        self.itemlist_urm = None
        self.ratinglist_urm = None

        # Splitting URM ----------------
        self.URM_all = None
        self.URM_train = None
        self.URM_test = None
        self.URM_val = None

        # ICM_subclass -----------------
        self.ICM = None
        self.itemlist_icm = None
        self.attributelist_icm = None
        self.presencelist_icm = None

        # ICM_asset --------------------
        self.ICM_asset = None
        self.itemlist_icm_asset = None
        self.assetlist_icm = None

        # ICM_price --------------------
        self.ICM_price = None
        self.itemlist_icm_price = None
        self.pricelist_icm = None

        # ICM_all ----------------------
        self.ICM_all = None

        # UCM_age ----------------------
        self.UCM_age = None
        self.userlist_ucm_age = None
        self.agelist_ucm = None
        self.presencelist_ucm_age = None

        # UCM_region -------------------
        self.UCM_region = None
        self.userlist_ucm_region = None
        self.regionlist_ucm = None
        self.presencelist_ucm_region = None

        # UCM_all ----------------------
        self.UCM_all = None

        # Tail-Boost -------------------
        self.dictionary = { 0 : [] }

    #######################################################################################
    #                         READING AND FORMATTING THE DATASET                          #
    #######################################################################################

    def rowSplit(self, rowString, token=","):
        split = rowString.split(token)
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)

        return result

    def get_file(self, file):
        return open("Data/" + file)

    def get_tuples(self, file, target=False):
        tuples = []
        file.seek(0)
        next(file)
        for line in file:
            if not target:
                tuples.append(self.rowSplit(line))
            if target:
                line = line.replace("\n", "")
                self.userlist_unique.append(int(line))
        return tuples

    def switch_source(self, argument):
        switcher = {
            1: "data_ICM_asset.csv",
            2: "data_ICM_price.csv",
            3: "data_ICM_sub_class.csv",
            4: "data_UCM_age.csv",
            5: "data_UCM_region.csv",
            6: "data_train.csv",
            7: "data_target_users_test.csv",
        }
        return switcher.get(argument, "Invalid argument")

    def get_list(self, tuples, matrix_type, source = None):
        firstlist, secondlist, thirdlist = zip(*tuples)

        if matrix_type == "URM":
            self.userlist_urm = list(firstlist)
            self.itemlist_urm = list(secondlist)
            self.ratinglist_urm = list(thirdlist)

        elif matrix_type == "ICM":
            if source == "subclass":
                self.itemlist_icm = list(firstlist)
                self.attributelist_icm = list(secondlist)
                self.presencelist_icm = list(thirdlist)

            if source == "asset":
                self.itemlist_icm_asset = list(firstlist)
                self.assetlist_icm = list(thirdlist)

            if source == "price":
                self.itemlist_icm_price = list(firstlist)
                self.pricelist_icm = list(thirdlist)

        elif matrix_type == "UCM":
            if source == "age":
                self.userlist_ucm_age = list(firstlist)
                self.agelist_ucm = list(secondlist)
                self.presencelist_ucm_age = list(thirdlist)

            if source == "region":
                self.userlist_ucm_region = list(firstlist)
                self.regionlist_ucm = list(secondlist)
                self.presencelist_ucm_region = list(thirdlist)

    def get_target_users(self):
        file = self.get_file(self.switch_source(7))
        self.get_tuples(file, target=True)

    #######################################################################################
    #                           SPLITTING AND CREATION OF MATRIX                          #
    #######################################################################################

    def split_dataset_loo(self, URM_train):
        print('Using LeaveOneOut for cross-validation')
        urm = URM_train.tocsr()
        users_len = len(urm.indptr) - 1
        items_len = max(urm.indices) + 1
        urm_train = urm.copy()
        urm_val = np.zeros((users_len, items_len))
        for user_id in range(users_len):
            start_pos = urm_train.indptr[user_id]
            end_pos = urm_train.indptr[user_id + 1]
            user_profile = urm_train.indices[start_pos:end_pos]
            if user_profile.size > 0:
                item_id = np.random.choice(user_profile, 1)
                urm_train[user_id, item_id] = 0
                urm_val[user_id, item_id] = 1

        urm_val = (sps.coo_matrix(urm_val, dtype=int, shape=urm.shape)).tocsr()
        urm_train = (sps.coo_matrix(urm_train, dtype=int, shape=urm.shape)).tocsr()

        urm_val.eliminate_zeros()
        urm_train.eliminate_zeros()

        self.URM_train = urm_train
        self.URM_val = urm_val

    def split_80_20(self, percentage=0.8):
        np.random.seed(123)
        urm = self.URM_all.tocsr()

        # Count the number of relevant interaction between user and item
        numInteractions = urm.nnz
        self.ratinglist_urm = np.array(self.ratinglist_urm)
        self.userlist_urm = np.array(self.userlist_urm)
        self.itemlist_urm = np.array(self.itemlist_urm)

        # Create a random mask to put on URM
        train_mask = np.random.choice([True, False], numInteractions, p=[percentage, 1 - percentage])

        # Create URM_train with all list of original dataset randomly choosed
        urm_train = sps.coo_matrix((self.ratinglist_urm[train_mask], (self.userlist_urm[train_mask], self.itemlist_urm[train_mask]))).tocsr()

        # Create URM_test using the logical_not operator applied on the train_mask (to keep the remain part of dataset as test set)
        test_mask = np.logical_not(train_mask)
        urm_test = sps.coo_matrix((self.ratinglist_urm[test_mask], (self.userlist_urm[test_mask], self.itemlist_urm[test_mask]))).tocsr()

        urm_test.eliminate_zeros()
        urm_train.eliminate_zeros()

        self.URM_train = urm_train
        self.URM_test = urm_test

    def get_URM(self):
        self.get_list(self.get_tuples(self.get_file(self.switch_source(6))), "URM")
        self.URM_all = sps.coo_matrix((self.ratinglist_urm, (self.userlist_urm, self.itemlist_urm))).tocsr()

    def get_ICM(self):

        # Get list for all matrix
        self.get_list(self.get_tuples(self.get_file(self.switch_source(3)), False), "ICM", "subclass")
        self.get_list(self.get_tuples(self.get_file(self.switch_source(2)), False), "ICM", "price")
        self.get_list(self.get_tuples(self.get_file(self.switch_source(1)), False), "ICM", "asset")

        # Pre-Processing
        le_asset = preprocessing.LabelEncoder()
        le_asset.fit(self.assetlist_icm)
        self.assetlist_icm = le_asset.transform(self.assetlist_icm)

        le_price = preprocessing.LabelEncoder()
        le_price.fit(self.pricelist_icm)
        self.pricelist_icm = le_price.transform(self.pricelist_icm)

        # Shaping
        n_items = self.URM_all.shape[1]
        n_features_icm_asset = max(self.assetlist_icm) + 1
        n_features_icm_price = max(self.pricelist_icm) + 1
        n_features_icm_sub_class = max(self.attributelist_icm) +1

        icm_asset_shape = (n_items, n_features_icm_asset)
        icm_price_shape = (n_items, n_features_icm_price)
        icm_sub_class_shape = (n_items, n_features_icm_sub_class)

        # Get matrix
        self.ICM_subclass = sps.coo_matrix((self.presencelist_icm, (self.itemlist_icm, self.attributelist_icm)), shape=icm_sub_class_shape)

        ones = np.ones(len(self.pricelist_icm))
        self.ICM_price = (sps.coo_matrix((ones, (self.itemlist_icm_price, self.pricelist_icm)), shape=icm_price_shape))

        ones = np.ones(len(self.assetlist_icm))
        self.ICM_asset = (sps.coo_matrix((ones, (self.itemlist_icm_asset, self.assetlist_icm)), shape=icm_asset_shape))

        self.ICM_all = sps.hstack((self.ICM_asset, self.ICM_price))
        self.ICM_all = sps.hstack((self.ICM_all, self.ICM_subclass))
        self.ICM_all = self.ICM_all.tocsr()

    def get_UCM(self):

        # UCM_region
        self.get_list(self.get_tuples(self.get_file(self.switch_source(5)), False), "UCM", "region")
        self.UCM_region = sps.coo_matrix((self.presencelist_ucm_region, (self.userlist_ucm_region, self.regionlist_ucm)))

        # UCM_age
        self.get_list(self.get_tuples(self.get_file(self.switch_source(4)), False), "UCM", "age")
        self.UCM_age = sps.coo_matrix((self.presencelist_ucm_age, (self.userlist_ucm_age, self.agelist_ucm)))

        self.UCM_all = sps.hstack((self.UCM_age, self.UCM_region))
        self.UCM_all = self.UCM_all.tocsr()

    #######################################################################################
    #                                  RECOMMENDER UTILS                                  #
    #######################################################################################

    def okapi_BM_25(self,dataMatrix, K1=1.2, B=0.75):

        assert B > 0 and B < 1, "okapi_BM_25: B must be in (0,1)"
        assert K1 > 0, "okapi_BM_25: K1 must be > 0"
        assert np.all(np.isfinite(dataMatrix.data)), \
            "okapi_BM_25: Data matrix contains {} non finite values".format(
                np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        # Weighs each row of a sparse matrix by OkapiBM25 weighting. Calculate idf per term (user)
        dataMatrix = sps.coo_matrix(dataMatrix)

        N = float(dataMatrix.shape[0])
        idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

        # calculate length_norm per document
        row_sums = np.ravel(dataMatrix.sum(axis=1))

        average_length = row_sums.mean()
        length_norm = (1.0 - B) + B * row_sums / average_length

        # weight matrix rows by bm25
        denominator = K1 * length_norm[dataMatrix.row] + dataMatrix.data
        denominator[denominator == 0.0] += 1e-9

        dataMatrix.data = dataMatrix.data * (K1 + 1.0) / denominator * idf[dataMatrix.col]
        return dataMatrix.tocsr()

    def TF_IDF(self, dataMatrix):

        assert np.all(np.isfinite(dataMatrix.data)), \
            "TF_IDF: Data matrix contains {} non finite values.".format(
                np.sum(np.logical_not(np.isfinite(dataMatrix.data))))

        assert np.all(dataMatrix.data >= 0.0), \
            "TF_IDF: Data matrix contains {} negative values, computing the square root is not possible.".format(
                np.sum(dataMatrix.data < 0.0))

        # TFIDF each row of a sparse amtrix
        dataMatrix = sps.coo_matrix(dataMatrix)
        N = float(dataMatrix.shape[0])

        # calculate IDF
        idf = np.log(N / (1 + np.bincount(dataMatrix.col)))

        # apply TF-IDF adjustment
        dataMatrix.data = np.sqrt(dataMatrix.data) * idf[dataMatrix.col]

        return dataMatrix.tocsr()

    def check_matrix(self, X, format='csc', dtype=np.float32):
        """
        This function takes a matrix as input and transforms it into the specified format.
        The matrix in input can be either sparse or ndarray.
        If the matrix in input has already the desired format, it is returned as-is
        the dtype parameter is always applied and the default is np.float32
        :param X:
        :param format:
        :param dtype:
        :return:
        """

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

    def feature_weight(self, matrix, feature_weighting):
        if feature_weighting == "BM25":
            matrix = matrix.astype(np.float32)
            matrix = self.okapi_BM_25(matrix.T).T
            matrix = self.check_matrix(matrix, 'csr')

        elif feature_weighting == "TF-IDF":
            matrix = matrix.astype(np.float32)
            matrix = self.TF_IDF(matrix.T).T
            matrix = self.check_matrix(matrix, 'csr')

        return matrix

    def score_dictionary(self, summed_score, user_id):
        if summed_score in self.dictionary.keys(): self.dictionary[summed_score].append(user_id)
        else:
            list = []
            list.append(user_id)
            self.dictionary[summed_score] = list

    #######################################################################################
    #                               SIMILARITY UTILS                                      #
    #######################################################################################

    def return_path(self):
        return os.getcwd()

    def export_similarity_matrix(self, filename, matrix, name=None):
        if name is not None:
            print("Fitting {0:s}...".format(name))
        sps.save_npz(filename, matrix)

    def export_nparr(self, filename, nparray):
        np.savez_compressed(self.return_path() + filename, nparray)

    def import_similarity_matrix(self, filename):
        return sps.load_npz(filename)

    def import_nparr(self, filename):
        dict_data = np.load(self.return_path() + filename)
        return dict_data['arr_0']

    def get_cosine_similarity_stored(self, matrix, name, SIMILARITY_PATH, knn, shrink, similarity, normalize, transpose=False, tuning=False):

        if transpose:
            matrix = matrix.T

        if not tuning:
            similarity_object = Compute_Similarity_Cython(matrix, shrink=shrink, topK=knn, normalize=normalize, similarity=similarity)
            W_sparse = similarity_object.compute_similarity()

        else:
            if not os.path.exists(self.return_path() + SIMILARITY_PATH):
                print("Fitting {0:s}...".format(name))
                similarity_object = Compute_Similarity_Cython(matrix, shrink=shrink, topK=knn, normalize=normalize, similarity=similarity)
                W_sparse = similarity_object.compute_similarity()
                self.export_similarity_matrix(self.return_path() + SIMILARITY_PATH, W_sparse)

            W_sparse = self.import_similarity_matrix(self.return_path() + SIMILARITY_PATH)

        return W_sparse

    def get_cosine_similarity(self, matrix, knn, shrink, similarity, normalize, transpose):
        if transpose:
            matrix = matrix.T

        similarity_object = Compute_Similarity_Cython(matrix, shrink=shrink, topK=knn, normalize=normalize, similarity=similarity)
        W_sparse = similarity_object.compute_similarity()
        return W_sparse




