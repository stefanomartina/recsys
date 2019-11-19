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


class Runner:
    def __init__(self, recommender, name, evaluate=True):
        self.recommender = recommender
        self.evaluate = evaluate
        self.userlist_unique = None
        self.URM_all = None
        self.name = name
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


    def get_URM_all(self):
        URM_file = self.get_URM_file()
        URM_tuples = self.get_URM_tuples(URM_file)

        userlist, itemlist, ratings = zip(*URM_tuples)
        self.userlist_unique = sorted(set(userlist))

        userList = list(userlist)
        itemList = list(itemlist)
        ratings = list(ratings)

        self.URM_all = sps.coo_matrix((ratings, (userList, itemList))).tocsr()

    def fit_recommender(self):
        print("fitting model")
        if not self.evaluate:
            self.recommender.fit(self.URM_all)
        else:
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

        #i have to call also the hold out function
        self.run_recommendations()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('recommender', choices=['random', 'topk', 'ItemCBF'], help="recomender?")
    parser.add_argument('--eval', help="would you evaluate?")
    args = parser.parse_args()

    recommender = None

    if args.recommender == 'random':
        print("random selected")
        recommender = RandomRecommender.RandomRecommender()

    if args.recommender == 'topk':
        print("topk selected")
        recommender = TopPopRecommender.TopPopRecommender()

    if args.recommender == 'ItemCBF':
        print("ItemCBF selected")
        # Dobbiamo passare al costruttore URM e ICM
        # DOMANI!!
        recommender = ItemCBFKNNRecommender.ItemCBFKNNRecommender()

    Runner(recommender, args.recommender, evaluate=False).run()
