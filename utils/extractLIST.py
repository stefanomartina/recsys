import zipfile
import scipy.sparse as sps


"""
    Function used for split each row of the dataset
    PARAM:  rowString - a single row of the dataset
            token - char which the function seeks to apply the split 
            n - number of column of the dataset
    
    RETURN: result - row splitted according token       
"""

def rowSplit(rowString, token, n):
    split = rowString.split(token)
    split[n - 1] = split[n - 1].replace("\n", "")
    for col in range(0, n):

        # rating must be a float
        if col == 2:
            split[col] = float(split[col])
        else:
            split[col] = int(split[col])
    result = tuple(split)
    return result


"""
    Function used to build URM matrix
    PARAM:  data_path - path of the zip file to be extracted
            dest_path - path in which save the extracted dataset 
            file_name - which part of the dataset to be extracted

    RETURN: formatted list extracted from dataset       
"""

def extractList(data_path, file_name, dest_path):
    dataFile = zipfile.ZipFile(data_path)
    matrix_path = dataFile.extract(file_name, path=dest_path)
    matrix_file = open(matrix_path, 'r')

    matrix_file.seek(0)
    matrix_tuples = []
    token = ","

    for line in matrix_file:
        if line != "row,col,data\n": matrix_tuples.append(rowSplit(line, token, 3))

    userList, itemList, ratingList = zip(*matrix_tuples)

    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)

    return [userList, itemList, ratingList]


""" 
    Function that create URM
    PARAM:  userList - list of users
            itemList - list of item 
            ratinList - list of ratings

    RETURN: formatted URM
    
"""
def createURM(userList, itemList, ratingList):
    # create the URM Matrix
    URM_all = sps.coo_matrix((ratingList, (userList, itemList))).tocsr()

    return URM_all


""" 
    Function that create ICM 
    PARAM:  URM_all - URM matrix 
            itemList - list of items
            classList - list of classes 
            presence - list of presence of the attributes

    RETURN: formatted ICM   
"""
def createICM(URM_all, itemlist, classList, presence):
    # define the number of item (with at least one interaction) nad number of attributes
    n_items = URM_all.shape[1]
    n_tags = max(classList) + 1

    ICM_shape = (n_items, n_tags)
    ICM_all = sps.coo_matrix((presence, (itemlist, classList)), shape=ICM_shape).tocsr()

    return ICM_all
