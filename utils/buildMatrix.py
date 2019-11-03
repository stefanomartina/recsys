import zipfile


# PARAMS:
#   - rowString: line of the dataset to be formatted
#   - URM : boolean used to understand which kind of matrix would be created (default: True = URM)

def row_split(row_string, urm=True):
    split = row_string.split("::")
    split[3] = split[3].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2]) if urm else str(split[2])
    split[3] = int(split[3])

    result = tuple(split)
    return result


# PARAMS:
#   - path : path from which dataset would be extracted
#   - wte : Which part to be extracted
#   - dest_path: destination path

def build(path, wte, dest_path, urm):
    datafile = zipfile.ZipFile(path)
    urm_path = datafile.extract(wte, path=dest_path)
    urm_file = open(urm_path, 'r')

    urm_file.seek(0)
    urm_tuples = []

    for line in urm_file:
        urm_tuples.append(row_split(line, urm))

    userList, itemList, ratingList, timestampList = zip(*urm_tuples)

    userList = list(userList)
    itemList = list(itemList)
    ratingList = list(ratingList)
    timestampList = list(timestampList)

    return [userList, itemList, ratingList, timestampList]
