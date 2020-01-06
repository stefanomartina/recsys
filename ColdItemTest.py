import os
import re
from collections import Counter

import numpy as np
import pandas as pd

from Base.BaseFunction import BaseFunction

filename = os.path.join(os.getcwd(), "Results/Hybrid-2020-01-05_00.36.20.csv")
def load_sample():
    cols = ['user_id', 'item_list']
    sample_data = pd.read_csv(filename, names=cols, header=0)
    return sample_data

if __name__ == "__main__":
    h = BaseFunction()
    h.get_URM()
    s = load_sample()
    x = s.item_list.values
    it=[]

    for i in x:
        it.append(re.findall(r'\d+', i))
    flattened = []

    for sublist in it:
        for val in sublist:
            flattened.append(int(val))

    item_pop = np.ediff1d(h.URM_all.tocsc().indptr)
    coldi = list(np.where(item_pop ==0)[0])

    z=Counter(flattened)
    tot=0
    for item in coldi:
        a=z[item]
        tot = tot +a
    print(tot)