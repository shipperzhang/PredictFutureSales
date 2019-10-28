import os
import re
import gzip
import datetime
import numpy as np
import statistics
import xgboost as xgb

class DataSet():
    def __init__(self):
        self.shops = None
        self.item_categories = None
        self.items = None
        self.prices = {}
        self.price_model = None
        self.priceX = None
        self.priceY = None

    def loadData(self, path):
        if path.endswith('.gz'):
            fopen = gzip.open
        else: fopen = open
        data = []
        with fopen(path, 'r') as f:
            for line in f.readlines():
                if isinstance(line, bytes): line = line.decode('utf-8')
                data.append(line.strip())
        return data

    def transfer2Vec(self, words, f):
        terms = {}
        counts = []
        for word in words:
            terms_in_word = {}
            pword = re.sub(r'[\[\],\"\(\)\.\-:]',' ',word)
            for c in pword.split():
                if len(c) > 0: 
                    if terms.get(c,None)==None: terms[c] = 0
                    terms[c] += 1
                    if terms_in_word.get(c,None)==None: terms_in_word[c] = 0
                    terms_in_word[c] += 1
            counts.append(terms_in_word)

        terms = [t for t in terms.keys() if terms[t] > f]
        n = len(terms)
        vectors = []
        for count in counts:
            vector = [0.0] * n
            for word in count.keys():
                if word in terms:
                    vector[terms.index(word)] = float(count[word])
            vectors.append(vector)
        # print(n)
        # print(len([v for v in vectors if (all(np.array(v) <= 0.0)==True)]))
        return vectors


    def loadInfo(self):
        self.shops = self.loadData('data/shops.csv')
        self.shops = [s.rsplit(',',1)[0][1:-1].replace('\"\"','\"') for s in self.shops[1:]]
        vectors = self.transfer2Vec(self.shops, 1)
        for i in range(len(self.shops)): self.shops[i] = [self.shops[i], vectors[i]]
            
        self.item_categories = self.loadData('data/item_categories.csv')
        self.item_categories = [s.rsplit(',',1)[0] for s in self.item_categories[1:]]
        vectors = self.transfer2Vec(self.item_categories, 1)
        for i in range(len(self.item_categories)): self.item_categories[i] = [self.item_categories[i], vectors[i]]

        self.items = self.loadData('data/items.csv')
        self.items = [[s.rsplit(',',2)[0].replace('\"',''), int(s.rsplit(',',2)[2])] for s in self.items[1:]]
        vectors = self.transfer2Vec([i[0] for i in self.items], 79)
        for i in range(len(self.items)): self.items[i] += [vectors[i]]


    def loadTrainData(self):
        trainX = []
        trainY = []
        if os.path.exists('trainDataFeatures.npy') and os.path.exists('trainDataLabel.npy') and False:
            trainX = np.load('trainDataFeatures.npy')
            trainY = np.load('trainDataLabel.npy')
        else:
            rawData = self.loadData('data/sales_train.csv.gz')
            date_blocks = []
            for _ in range(34): date_blocks.append({})
            for data in rawData[1:]:
                units = data.split(',')
                date = [int(u) for u in units[0].split('.')]
                dt = datetime.datetime(date[2],date[1],date[0]).year
                block_num = int(units[1])
                shop_id = int(units[2])
                item_id = int(units[3])
                item_price = float(units[4])
                item_cnt_day = float(units[5])
                if item_cnt_day < 0.0: continue
                key = str(shop_id) + ',' + str(item_id)
                if self.prices.get(key, None)==None or self.prices[key][1] < dt:
                    self.prices[key] = [item_price, dt]
                if date_blocks[block_num].get(key, None)==None:
                    date_blocks[block_num][key] = [[item_price],item_cnt_day]
                else: 
                    date_blocks[block_num][key][0].append(item_price)
                    date_blocks[block_num][key][1] += item_cnt_day
            for i in range(34):
                month = float(1 + ( i % 12 ))
                year = float(2013 + ( i // 12))
                for key in date_blocks[i].keys():
                    features = []
                    shop_id = int(key.split(',')[0])
                    item_id = int(key.split(',')[1])
                    category_id = self.items[item_id][1]
                    item_price = statistics.mean(date_blocks[i][key][0])
                    item_cnt_month = date_blocks[i][key][1]
                    features += [float(shop_id), float(item_id), float(category_id), float(year), float(month), item_price]
                    features += self.shops[shop_id][1]
                    features += self.items[item_id][2]
                    features += self.item_categories[category_id][1]
                    trainX.append(np.array(features))
                    trainY.append(item_cnt_month)
            trainX = np.array(trainX)
            trainY = np.array(trainY)
            self.priceX = np.delete(trainX, 5, 1)
            self.priceY = trainX[:,5]
            model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, \
                subsample = 1, eta = 0.2, num_round = 1000, seed = 1)
            print("Train Price Model ...")
            self.price_model = model.fit(self.priceX, self.priceY)
            print("Done.")
            np.save('trainDataFeatures.npy', trainX)
            np.save('trainDataLabel.npy', trainY)
        print(np.shape(trainX)[:2])
        return trainX, trainY
            
    def loadTestData(self):
        testX = []
        if os.path.exists('testDataFeatures.npy') and False:
            testX = np.load('testDataFeatures.npy')
        else:
            rawData = self.loadData('data/test.csv.gz')
            month = 11
            year = 2015
            for data in rawData[1:]:
                features = []
                units = data.split(',')
                # ID = int(units[0])
                shop_id = int(units[1])
                item_id = int(units[2])
                category_id = self.items[item_id][1]
                features += [float(shop_id), float(item_id), float(category_id), float(year), float(month), 0.0]
                features += self.shops[shop_id][1]
                features += self.items[item_id][2]
                features += self.item_categories[category_id][1]
                key = str(shop_id) + ',' + str(item_id)
                if self.prices.get(key,None) == None:
                    testP = np.array([np.delete(features, 5)])
                    self.prices[key] = [self.price_model.predict(testP)[0],None]
                    # if ID < 100: print(key + ',' + str(self.prices[key][0]))
                item_price = self.prices[key][0]
                features[5] = item_price
                testX.append(np.array(features))
            testX = np.array(testX)
            np.save('testDataFeatures.npy', testX)
        print(np.shape(testX)[:2])
        return testX

if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    dataset = DataSet()
    dataset.loadInfo()
    dataset.loadTrainData()
    dataset.loadTestData()
