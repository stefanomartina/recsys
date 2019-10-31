import csv
import numpy as np


# PARAMS:
#   - file_path: path of the file
# RETURNED VALUE
#   - row : a row containing the entire dataset
def open_csv(file_path):
    row = []
    with open(file_path) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        # csv_read = csv.reader(csv_file)   ---> Delimiter not needed??
        for element in csv_read:
            row.append(element)
    return row


# PARAMS:
#   - waw: what to write
#   - file_path: path of the file
def write_csv(file_path, fields, rows):
    with open(file_path, 'w') as csv_file:
        csv_write = csv.writer(csv_file)
        csv_write.writerow(fields)
        csv_write.writerows(rows)


def row_to_matrix(row):
    return np.matrix(row)


def tuples_from_matrix(matrix):
    row = []
    n_of_columns = np.size(matrix, 1)

    for i in range(n_of_columns):
        # matrix[:,i] --> i-th column of the matrix
        # added as i-th element of the list of tuple
        row.insert(i, matrix[:, i])

    return row


if __name__ == '__main__':
    local_path = "/Users/Stefano/Downloads/recommender-system-2018-challenge-polimi/"
    relative_path = "train.csv"
    complete_path = local_path + relative_path

    row = open_csv(local_path + relative_path)
    print(row)

    matrix = row_to_matrix(row)
    first, second = tuples_from_matrix(matrix)