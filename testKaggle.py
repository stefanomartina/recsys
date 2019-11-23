import zipfile

import numpy as np
import pandas as pd
from sklearn import preprocessing
from utils import extractCSV as exc
from utils import extractLIST as exl
from utils import splitDataset as spl
import Runner as r
from recommenders import RandomRecommender as rr
from recommenders import TopPopRecommender as tp
from recommenders import createRecommendation as createRec
from recommenders import ItemCBFKNNRecommender as knn
import os
import scipy.sparse as sps

recommender = tp.TopPopRecommender()
name = None
runner = r.Runner(recommender,name)
runner.get_ICM_merged()



