import zipfile

def rowSplit (rowString, URM=True):
    split = rowString.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2]) if URM else str(split[2])
    split[3] = int(split[3])

    result = tuple(split)
    return result


# path : path of the to be extracted file
# wte : Which part to be extracted
# dest_path: destination path

def build(path, wte, dest_path, URM):
    dataFile = zipfile.ZipFile(path)
    URM_path = dataFile.extract(wte, path= dest_path)
    URM_file = open(URM_path, 'r')

    URM_file.seek(0)
    URM_tuples = []

    for line in URM_file:
        URM_tuples.append(rowSplit(line, URM))

    userList, itemList, ratingList, timestampList = zip(*URM_tuples)

    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)
    timestampList = list(timestampList)

    return [userList, itemList, ratingList, timestampList]