import numpy as np


def list_ID_stats(ID_list, label):
    min_val = min(ID_list)
    max_val = max(ID_list)
    unique_value = len(set(ID_list))
    missing_value = 1 - unique_value / (max_val - min_val)

    print("{} data, ID: min {}, max {}, unique {}, missing {:.2f} %".format(label, min_val, max_val, unique_value, missing_value * 100))


""" 
    Function that calculate the item popularity in URM    
    PARAM:      URM_all - representazione of interaction between user and item
                at - number of item to be displayed 
    RETURN:     simple print
"""

def item_popularity(URM_all, at, reverse_value):
    itemPopularity = (URM_all > 0).sum(axis=0)
    itemPopularity = np.array(itemPopularity).squeeze()
    itemPopularity = np.sort(itemPopularity, reverse = reverse_value)
    print(itemPopularity)