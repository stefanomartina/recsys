from utils import extractCSV as csv
from utils import splitDataset as spl
import zipfile
import numpy as np

""" This main function is used to test various functions"""

# Path Dataset Stefano: /Users/Stefano/Downloads/recommender-system-2018-challenge-polimi/
# Path Dataset Simone: D:\Download\recommender-system-2018-challenge-polimi\

if __name__ == '__main__':

    """
    local_path = input("Enter path of dataset: ")
    relative_path = "train.csv"
    complete_path = local_path + relative_path

    row = csv.open_csv(local_path + relative_path)

    matrix = csv.row_to_matrix(row)
    first, second = csv.tuples_from_matrix(matrix)
    print(first[:10])
    print(second[:10])
    """

    URM_tuples = [[34,67,8],[7,12,34],[78,865,45],[789,9,87],[5,9,7],[89,52,63],[12,54,23],[22,28,79],[4,5,3],[2,1,3],[3,5,4]]
    URM_test, URM_train = spl.splitDataset(0.80, URM_tuples)
    print("-------")
    print(URM_train)
    print("-------")
    print(URM_test)
