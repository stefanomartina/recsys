import zipfile

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
    URM_path = dataFile.extract(file_name, path=dest_path)
    URM_file = open(URM_path, 'r')

    URM_file.seek(0)
    URM_tuples = []
    token = ","

    for line in URM_file:
        if line != "row,col,data\n": URM_tuples.append(rowSplit(line, token, 3))

    userList, itemList, ratingList = zip(*URM_tuples)

    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)

    return [userList, itemList, ratingList]
