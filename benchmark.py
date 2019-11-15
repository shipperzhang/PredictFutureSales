import numpy as np
import pandas as pd

import os

train = pd.read_csv('./data/sales_train.csv.gz')
test = pd.read_csv('./data/test.csv.gz')
submission = pd.read_csv('./data/sample_submission.csv.gz')
items = pd.read_csv('./data/items.csv')
item_cats = pd.read_csv('./data/item_categories.csv')
shops = pd.read_csv('./data/shops.csv')

print(train.describe())