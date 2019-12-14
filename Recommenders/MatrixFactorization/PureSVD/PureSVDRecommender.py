""" Created on 14/06/18 @author: Maurizio Ferrari Dacrema """
import os

import numpy as np
import scipy.sparse as sps
from scipy import sparse

from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD

SVD_LIBRARY = ["svd", "svds", "randomized_svd", "Trunked_svd"]

class PureSVDRecommender(object):

    def fit(self, URM_train, verbose=True, library="random_svd",  n_components =200, n_iter=5, num_factors=2000):
        self.URM = URM_train
        self.vebose = verbose

        if library == "random_svd":
            self.get_URM_Random_SVD(n_components=n_components, n_iter=n_iter)

        if library == "svds":
            self.get_URM_SVDS(num_factors=num_factors)

        self.similarityProduct = self.U.dot(self.Sigma_Vt)

    def get_expected_ratings(self, user_id):
        expected_scores = (self.similarityProduct[user_id]).ravel()
        return expected_scores

    def recommend(self, user_id, at=10):
        expected_ratings = self.get_expected_ratings(user_id)
        recommended_items = np.flip(np.argsort(expected_ratings), 0)

        unseen_items_mask = np.in1d(recommended_items, self.URM[user_id].indices, assume_unique=True, invert=True)
        recommended_items = recommended_items[unseen_items_mask]
        return recommended_items[0:at]

    def get_URM_SVDS(self, num_factors=2000, which='LM'):
        print("Computing SVD decomposition...")

        U, Sigma, VT = svds(self.URM, k=num_factors, which=which)
        UT = U.T
        Sigma_2_flatten = np.power(Sigma, 2)
        Sigma_2 = np.diagflat(Sigma_2_flatten)
        Sigma_2_csr = sparse.csr_matrix(Sigma_2)
        URM_SVDS = U.dot(Sigma_2_csr.dot(UT))
        print("Computing SVD decomposition... Done!")

        return URM_SVDS

    def get_URM_Random_SVD(self, n_components =200, n_iter=5, random_seed=None):
        print("Computing SVD decomposition...")

        # U: 30911x2000
        # SIGMA: 2000
        # VT: 2000x18495
        self.U, Sigma, VT = randomized_svd(self.URM, n_components = n_components, n_iter=n_iter, random_state=random_seed)

        # SIGMA_VT: 2000x18495
        self.Sigma_Vt = sps.diags(Sigma)*VT

        print("Computing SVD decomposition... Done!")
