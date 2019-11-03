import csv
import numpy as np

""" Script to extract and format a file in CSV format"""

# PARAMS:
#   - file_path: path of the file
# RETURNED VALUE
#   - row : a row containing the entire dataset as a tuple
def open_csv(file_path):
    row = []
    with open(file_path) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        # csv_read = csv.reader(csv_file)   ---> Delimiter not needed??
        for element in csv_read:
            row.append(element)
    return row


# PARAMS:
#   - file_path: path of the file
def write_csv(file_path, fields, rows):
    with open(file_path, 'w') as csv_file:
        csv_write = csv.writer(csv_file)
        csv_write.writerow(fields)
        csv_write.writerows(rows)


# PARAMS:
#   - row : tuple to be converted into a matrix
def row_to_matrix(row):
    return np.matrix(row)


# PARAMS:
#   - matrix : this is a smart rapresentation of dataset, from which will be extracted all
#              users' list, items' list and so on.
def tuples_from_matrix(matrix):
    row = []
    n_of_columns = np.size(matrix, 1)

    for i in range(n_of_columns):
        # matrix[:,i] --> i-th column of the matrix
        # added as i-th element of the list of tuple
        row.insert(i, matrix[:, i])

    return row