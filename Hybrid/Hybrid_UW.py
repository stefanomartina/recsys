from Hybrid.BaseHybridRecommender import BaseHybridRecommender
import numpy as np



item_cf_param = {
    "knn": 10,
    "shrink": 30,
}

user_cf_param = {
    "knn": 600,
    "shrink": 0,
}

item_cb_param = {
    "knn": 5,
    "shrink": 0,
}

slim_param = {
    "epochs": 200,
    "topK": 10,
}


class Hybrid_User_Wise(BaseHybridRecommender):

    #######################################################################################
    #                                 FITTING ALGORITHM                                   #
    #######################################################################################

    def fit(self, URM, ICM_all=None, UCM_all=None,
                   knn_itemcf=item_cf_param["knn"], shrink_itemcf=item_cf_param["shrink"],
                   knn_usercf=user_cf_param["knn"], shrink_usercf=item_cf_param["shrink"],
                   knn_itemcb=item_cb_param["knn"], shrink_itemcb=item_cb_param["shrink"],tuning=False,
            thre1=1, thre2=1.5, thre3=2.5, thre4=3, thre5=4.4):

        self.URM = URM
        self.ICM_all = ICM_all
        self.UCM_all = UCM_all
        self.thre1 = thre1
        self.thre2 = thre2
        self.thre3 = thre3
        self.thre4 = thre4
        self.thre5 = thre5


        # Sub-Fitting
        self.itemCF.fit(URM.copy(), knn_itemcf, shrink_itemcf, tuning=tuning)
        self.userCF.fit(URM.copy(), knn_usercf, shrink_usercf, tuning=tuning)
        self.userContentBased.fit(URM.copy(), UCM_all, tuning=tuning)
        self.itemContentBased.fit(URM.copy(), ICM_all, knn_itemcb, shrink_itemcb, tuning=tuning)
        self.slim_random.fit(URM.copy(), tuning=tuning)
        self.RP3Beta.fit(URM.copy(), tuning=tuning)


    #######################################################################################
    #                                  EXTRACT RATINGS                                    #
    #######################################################################################

    def get_user_profile_length(self, user_id):
        #return self.URM[user_id].nnz
        return np.ediff1d(self.URM.indptr)[user_id]

    def recommend(self, user_id, at=10):
        l = self.get_user_profile_length(user_id)
        if l < self.thre1:
            return self.userContentBased.recommend(user_id)
        elif l < self.thre2:
            return self.userCF.recommend(user_id)
        elif l < self.thre3:
            return self.itemCF.recommend(user_id)
        elif l < self.thre4:
            return self.itemContentBased.recommend(user_id)
        elif l < self.thre5:
            return self.itemCF.recommend(user_id)
        else:
            return self.RP3Beta.recommend(user_id)
