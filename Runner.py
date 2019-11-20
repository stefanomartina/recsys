import argparse
from recommenders import RandomRecommender
from recommenders import TopPopRecommender
from recommenders import ItemCBFKNNRecommender
from utils import evaluation
from utils import splitDataset
from utils import extractCSV
from utils import extractLIST
import os
import zipfile
import scipy.sparse as sps
from utils import extractCSV
import numpy as np



class Runner:
    def __init__(self, recommender, name, evaluate=True):
        print("Evaluate: " + str(evaluate))
        self.recommender = recommender
        self.evaluate = evaluate
        self.userlist_unique = None
        self.name = name

        self.URM_all = None

        self.validation_mask = None
        self.train_mask = None

        self.userlist = None
        self.itemlist = None
        self.ratinglist = None

        # NOT YET IMPLEMENTED
        self.URM_train = None
        self.URM_validation = None

    def rowSplit(self, rowString, token=","):
        split = rowString.split(token)
        split[2] = split[2].replace("\n", "")

        split[0] = int(split[0])
        split[1] = int(split[1])
        split[2] = float(split[2])

        result = tuple(split)
        return result

    def get_URM_file(self, relative_path="/data/recommender-system-2019-challenge-polimi.zip", file="data_train.csv"):
        dirname = os.path.dirname(__file__)
        dataFile = zipfile.ZipFile(dirname + relative_path)
        URM_path = dataFile.extract(file, path=dirname + "/data")
        return open(URM_path, 'r')

    def get_URM_tuples(self, URM_file):
        URM_tuples = []
        URM_file.seek(0)

        for line in URM_file:
            if line != "row,col,data\n":
                URM_tuples.append(self.rowSplit(line))
        return URM_tuples

    def split_dataset(self):
        URM_shape = np.shape(self.URM_all)
        print("URM_SHAPE: " + str(URM_shape))

        tuple = [False] * URM_shape[0] + [True]

        self.train_mask = []
        print("Splitting dataset in train and validation...")
        for i in range(URM_shape[1]):
            np.random.shuffle(tuple)
            self.train_mask.append(tuple)

        self.validation_mask = np.logical_not(self.train_mask)

        self.URM_train = sps.coo_matrix((self.ratinglist[self.train_mask], (self.userlist[self.train_mask], self.itemlist[self.train_mask]))).tocsr()
        self.URM_validation = sps.coo_matrix((self.ratinglist[self.validation_mask], (self.userlist[self.validation_mask], self.itemlist[self.validation_mask]))).tocsr()
        print("Split completed")

    def get_URM_all(self):
        URM_file = self.get_URM_file()
        URM_tuples = self.get_URM_tuples(URM_file)

        userlist, itemlist, ratingslist = zip(*URM_tuples)
        self.userlist_unique = sorted(set(userlist))

        self.userlist = list(userlist)
        self.itemlist = list(itemlist)
        self.ratinglist = list(ratingslist)

        self.URM_all = sps.coo_matrix((self.ratinglist, (self.userlist, self.itemlist))).tocsr()

    def fit_recommender(self):
        print("fitting model")
        if not self.evaluate:
            self.recommender.fit(self.URM_all)
        else:
            self.split_dataset()
            self.recommender.fit(self.URM_train)
        print("Model fitted")

    def run_recommendations(self):
        recommendations = []
        saved_tuple = []
        print("Computing recommendations...")
        for user in self.userlist_unique:
            index = [str(user) + ","]
            recommendations.clear()

            for recommendation in self.recommender.recommend(user):
                recommendations.append(recommendation)
            saved_tuple.append(index + recommendations)
        print("Recommendations computed")
        print("Printing csv...")
        extractCSV.write_csv(saved_tuple, self.name)
        print("Ended - BYE BYE")

    def run(self):
        self.get_URM_all()
        self.fit_recommender()
        self.run_recommendations()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'top-pop', 'ItemCBF'])
    parser.add_argument('--eval', action="store_true")
    args = parser.parse_args()

    recommender = None



    if args.recommender == 'random':
        print("random selected")
        recommender = RandomRecommender.RandomRecommender()

    if args.recommender == 'top-pop':
        print("top-pop selected")
        recommender = TopPopRecommender.TopPopRecommender()

    if args.recommender == 'ItemCBF':
        print("ItemCBF selected")
        # Dobbiamo passare al costruttore URM e ICM
        # DOMANI!!
        recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()
    print(args)
    Runner(recommender, args.recommender, evaluate=args.eval).run()
