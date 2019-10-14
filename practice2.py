from urllib.request import urlretrieve
import zipfile

if __name__ == '__main__':
    urlretrieve("http://files.grouplens.org/datasets/movielens/ml-10m.zip", "/Users/Stefano/PycharmProjects/recsys/data/Movielens_10M/movielens_10m.zip")

    print("ciao")
    dataFile = zipfile.ZipFile("data/Movielens_10M/movielens_10m.zip")

    URM_path = dataFile.extract("ml-10M100K/ratings.dat", path= "data/Movielens_10M")

    URM_file = open(URM_path, 'r')

    type(URM_file)

    for _ in range(10):
        print(URM_file.readline())

