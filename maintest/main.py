from utils import extractCSV as csv

""" This main function is used to test various functions"""

# Path Dataset Stefano: /Users/Stefano/Downloads/recommender-system-2018-challenge-polimi/
# Path Dataset Simone: D:\Download\recommender-system-2018-challenge-polimi\

if __name__ == '__main__':

    local_path = input("Enter path of dataset: ")
    relative_path = "train.csv"
    complete_path = local_path + relative_path

    row = csv.open_csv(local_path + relative_path)

    matrix = csv.row_to_matrix(row)
    first, second = csv.tuples_from_matrix(matrix)
    print(first[:10])
    print(second[:10])