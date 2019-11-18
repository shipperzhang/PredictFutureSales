import os
import re
import gzip
import datetime
import numpy as np
import statistics
import xgboost as xgb
import pandas as pd

class DataSet():
    def __init__(self):
        self.shops = None
        self.item_categories = None
        self.items = None
        # save the latest price for each (shop_id, item_id) pair in training data
        self.prices = {}
        # train a price for the (shop_id, item_id) pairs without price in training data
        self.price_model = None
        self.priceX = None
        self.priceY = None

    def loadDataFromFile(self, path):
        """
        Load Data from file
        """
        if path.endswith('.gz'):
            fopen = gzip.open
        else: fopen = open
        data = []
        with fopen(path, 'r') as f:
            for line in f.readlines():
                if isinstance(line, bytes): line = line.decode('utf-8')
                data.append(line.strip())
        return data

    def transfer2Vec(self, words):
        """
        Transfer the words array to vectors array.
        """
        terms = set()
        counts = []
        for word in words:
            terms_in_word = {}
            pword = re.sub(r'[\[\],\"\(\)\.\-:]',' ',word)
            for c in pword.split():
                if len(c) > 0: 
                    terms.add(c)
                    terms_in_word[c] = '1'
            counts.append(terms_in_word)
        terms = list(terms)
        n = len(terms)
        vectors = []
        for count in counts:
            ir = ['0'] * n
            for word in count.keys():
                ir[terms.index(word)] = count[word]
            vector = []
            for i in range((n // 128) + 1):
                vector.append(float(int(''.join(ir[i*128:(i+1)*128]),2)))
            vectors.append(vector)
        return vectors


    def loadInfo(self):
        """
        Load basic information for shops, items & item_categories
        """
        self.shops = self.loadDataFromFile('data/shops.csv')
        self.shops = [s.rsplit(',',1)[0][1:-1].replace('\"\"','\"') for s in self.shops[1:]]
        vectors = self.transfer2Vec(self.shops)
        for i in range(len(self.shops)): self.shops[i] = [self.shops[i], vectors[i]]
            
        self.item_categories = self.loadDataFromFile('data/item_categories.csv')
        self.item_categories = [s.rsplit(',',1)[0] for s in self.item_categories[1:]]
        vectors = self.transfer2Vec(self.item_categories)
        for i in range(len(self.item_categories)): self.item_categories[i] = [self.item_categories[i], vectors[i]]

        self.items = self.loadDataFromFile('data/items.csv')
        self.items = [[s.rsplit(',',2)[0].replace('\"',''), int(s.rsplit(',',2)[2])] for s in self.items[1:]]
        vectors = self.transfer2Vec([i[0] for i in self.items])
        for i in range(len(self.items)): self.items[i] += [vectors[i]]


    def loadTrainData(self, reProcess=False):
        """
        Load Training Data
        """
        trainX = []
        trainY = []
        if os.path.exists('trainDataFeatures.npy') and os.path.exists('trainDataLabel.npy') and not reProcess:
            trainX = np.load('trainDataFeatures.npy')
            trainY = np.load('trainDataLabel.npy')
        else:
            self.loadInfo()
            testData = self.loadDataFromFile('data/test.csv.gz')
            pairsInTest = set()
            shopsInTest = set()
            for data in testData[1:]:
                shop_id = int(data.split(',')[1])
                item_id = int(data.split(',')[2])
                key = str(shop_id) + ',' + str(item_id)
                pairsInTest.add(key)
                shopsInTest.add(shop_id)

            train = pd.read_csv('./data/sales_train.csv.gz')
            train = train.loc[train['item_cnt_day'] < 3.0].loc[train['item_price'] < 1000.0]
            train = train.groupby(["date_block_num","shop_id", "item_id"])
            train = train.aggregate({"item_price":np.mean, "item_cnt_day":np.sum}).fillna(0)
            train.reset_index(level=["date_block_num", "shop_id", "item_id"], inplace=True)

            # sum up item_cnt_day to item_month_day and extract price information
            date_blocks = []
            for _ in range(34): date_blocks.append({})
            for _, data in train.iterrows():
                block_num = int(data['date_block_num'])
                shop_id = int(data['shop_id'])
                item_id = int(data['item_id'])
                item_price = data['item_price']
                item_cnt_month = data['item_cnt_day']
                key = str(shop_id) + ',' + str(item_id)
                if self.prices.get(key, None) == None or self.prices[key][1] < block_num:
                    self.prices[key] = [item_price, block_num]
                date_blocks[block_num][key] = [item_price, item_cnt_month]
            
            # generate features
            for i in range(34):
                month = 1 + ( i % 12 )
                year = 2013 + ( i // 12)
                pairsInTrain = set()
                shopsInTrain = set()
                itemsInTrain = set()
                for key in date_blocks[i].keys():
                    features = []
                    shop_id = int(key.split(',')[0])
                    item_id = int(key.split(',')[1])
                    category_id = self.items[item_id][1]
                    item_price = date_blocks[i][key][0]
                    item_cnt_month = date_blocks[i][key][1]
                    # if item_cnt_month < 0.0: continue
                    features += [shop_id, item_id, category_id]
                    # if year == 2015: features += [2015.0,0.0]
                    # if month == 11: features += [0.0,10.0]
                    features += [year, month]
                    features.append(item_price)
                    #features += self.shops[shop_id][1]
                    #features += self.items[item_id][2]
                    #features += self.item_categories[category_id][1]
                    pairsInTrain.add(key)
                    shopsInTrain.add(shop_id)
                    itemsInTrain.add(item_id)
                    trainX.append(np.array(features))
                    trainY.append(item_cnt_month)
                # if year == 2015 or month == 11:
                pairsNotInTrain = set()
                for key in pairsInTest.difference(pairsInTrain):
                    shop_id = int(key.split(',')[0])
                    item_id = int(key.split(',')[1])
                    if shop_id in shopsInTrain and item_id in itemsInTrain: 
                        pairsNotInTrain.add(key)
                ###print(len(pairsNotInTrain))
                for key in pairsNotInTrain:
                    features = []
                    shop_id = int(key.split(',')[0])
                    item_id = int(key.split(',')[1])
                    category_id = self.items[item_id][1]
                    features += [shop_id, item_id, category_id]
                    features += [year, month]
                    features.append(item_price)
                    #features += self.shops[shop_id][1]
                    #features += self.items[item_id][2]
                    #features += self.item_categories[category_id][1]
                    trainX.append(np.array(features))
                    trainY.append(0.0)
            
            trainX = np.array(trainX)
            trainY = np.array(trainY)
            """
            # Train the model for the price
            self.priceX = np.delete(trainX, 5, 1)
            self.priceY = trainX[:,5]
            model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, \
                subsample = 1, eta = 0.2, num_round = 1000, seed = 1)
            print("Train Price Model ...")
            self.price_model = model.fit(self.priceX, self.priceY)
            print("Done.")
            """

            # Save the data for the future convenience
            # np.save('trainDataFeatures.npy', trainX)
            # np.save('trainDataLabel.npy', trainY)

        ###print(np.shape(trainX)[:2])
        return trainX, trainY
            
    def loadTestData(self, reProcess=False):
        """
        Load Testing Data
        """
        testX = []
        if os.path.exists('testDataFeatures.npy') and not reProcess:
            testX = np.load('testDataFeatures.npy')
        else:
            rawData = self.loadDataFromFile('data/test.csv.gz')

            # generate feature for each test data
            month = 11
            year = 2015
            for data in rawData[1:]:
                features = []
                units = data.split(',')
                # ID = int(units[0])
                shop_id = int(units[1])
                item_id = int(units[2])
                category_id = self.items[item_id][1]
                features += [shop_id, item_id, category_id]
                features += [year, month]
                features.append(0.0)
                #features += self.shops[shop_id][1]
                #features += self.items[item_id][2]
                #features += self.item_categories[category_id][1]
                """
                key = str(shop_id) + ',' + str(item_id)
                if self.prices.get(key,None) == None:
                    # Get the price from model if there is no such a (shop_id, item_id) pair 
                    # in training data 
                    testP = np.array([np.delete(features, 5)])
                    self.prices[key] = [self.price_model.predict(testP)[0],None]
                    # if ID < 100: print(key + ',' + str(self.prices[key][0]))
                item_price = self.prices[key][0]
                features[5] = item_price
                """
                testX.append(np.array(features))
            testX = np.array(testX)

            # Save the data for the future convenience
            # np.save('testDataFeatures.npy', testX)
        ###print(np.shape(testX)[:2])
        return testX

if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    dataset = DataSet()
    trainX, trainY = dataset.loadTrainData(True)
    testX = dataset.loadTestData(True)
    print(trainX[0])
    print('\n')
    print(testX[0])

