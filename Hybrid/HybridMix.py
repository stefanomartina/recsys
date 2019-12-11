from Recommenders.Slim.SlimBPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Recommenders.Collaborative.ItemKNNCFRecommender import ItemKNNCFRecommender
import numpy as np
slim_param = {
    "epochs": 200,
    "topK": 10,
}

item_cf_param = {
    "knn": 10,
    "shrink": 30,
}

class HybridMix():
    def __init__(self):
        self.ItemCF = ItemKNNCFRecommender()
        self.Slim = SLIM_BPR_Cython()

        self.merge = None
        self.merge_index = 5

    def fit(self, URM):
        self.URM = URM

        self.ItemCF.fit(URM)
        self.Slim.fit(URM)

    def recommend(self, user_id, at=10):
        ItemCF_rec = self.ItemCF.recommend(user_id).tolist()
        Slim_rec = self.Slim.recommend(user_id).tolist()
        cont = 0

        for elem in Slim_rec[0:10-self.merge_index]:
            if elem not in ItemCF_rec:
                # Add element in ItemCF
                ItemCF_rec[self.merge_index+cont] = elem
            cont += 1

        return ItemCF_rec
