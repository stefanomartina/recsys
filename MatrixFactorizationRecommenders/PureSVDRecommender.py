""" Created on 14/06/18 @author: Maurizio Ferrari Dacrema """

from MatrixFactorizationRecommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from sklearn.utils.extmath import randomized_svd
import scipy.sparse as sps

class PureSVDRecommender(BaseMatrixFactorizationRecommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def fit(self, URM_train, verbose = True, num_factors=600, random_seed = None):
        super(PureSVDRecommender, self).instanziate_rec(URM_train, verbose)
        self._print("Computing SVD decomposition...")

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_components=num_factors,
                                      n_iter=5,
                                      random_state = random_seed)

        s_Vt = sps.diags(Sigma)*VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        self._print("Computing SVD decomposition... Done!")


