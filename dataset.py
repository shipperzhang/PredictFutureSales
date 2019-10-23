import os
import gzip
import numpy as np

def loadData(path):
    if path.endswith('.gz'):
        fopen = gzip.open
    else: fopen = open
    data = []
    with fopen(path, 'r') as f:
        for line in f.readlines():
            if isinstance(line, bytes): line = line.decode('utf-8')
            data.append(line.strip())
    return data

def loadTrainData():
    rawData = loadData('data/sales_train.csv.gz')
    items = loadData('data/items.csv')
    shops = loadData('data/shops.csv')
    item_categories = loadData('data/item_categories.csv')
    trainX = []
    trainY = []
    count = 0
    for d in rawData[1:]:
        features = []
        units = d.split(',')
        date = units[0]
        shop_id = int(units[2])
        item_id = int(units[3])
        item_price = float(units[4])
        item_cnt_day = float(units[5])
        if item_cnt_day < -1.0: print(d)
        trainX.append(features)
        trainY.append(item_cnt_day)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY
        
def loadTestData():
    pass

if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    loadTrainData()