import csv
import numpy as np


""" 
    Function to open a csv file 
        PARAM:    file_path - path of the file
        RETURN:   row - a row containing the entire dataset as a tuple
"""

def open_csv(file_path):
    row = []
    with open(file_path) as csv_file:
        next(csv_file)
        csv_read = csv.reader(csv_file, delimiter=',')
        for element in csv_read:
            row.append(element)
    return row


"""
    Function to write a csv file 
        PARAM:    file_path - path of the file
"""

def write_csv(file_path, fields, rows):
    with open(file_path, 'w') as csv_file:
        csv_write_head = csv.writer(csv_file, delimiter=',')
        csv_write_head.writerow(fields)
        csv_write_content = csv.writer(csv_file, delimiter=' ')
        csv_write_content.writerows(rows)


"""
    Function to write a csv file 
        PARAM:    row -  tuple to be converted into a matrix
"""

def row_to_matrix(row):
    return np.matrix(row)


"""
    Function to extrac all meaningfull list from matrix 
        PARAM:    matrix - a matrix from which estract tuples
"""

def tuples_from_matrix(matrix):
    row = []
    n_of_columns = np.size(matrix, 1)

    for i in range(n_of_columns):
        # matrix[:,i] --> i-th column of the matrix
        # added as i-th element of the list of tuple
        row.insert(i, matrix[:, i])

    return row