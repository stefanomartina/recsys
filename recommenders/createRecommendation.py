from utils import extractCSV as exc

def recommandations(userList_unique, URM_train, at=10):
    recommender = object
    recommender.fit(URM_train)

    appo = []
    index = []
    saved_tuple = []

    # create the result.csv
    for i in userList_unique:
        index = [i[0] + ","]
        appo.clear()
        for i in recommender.recommend(int(i[0]), at):
            appo.append(i)

        saved_tuple.append(index + appo)

    exc.write_csv("test.csv", ["user_id", "item_list"], saved_tuple)