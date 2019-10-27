import os
import re
import gzip
import datetime
import numpy as np

class DataSet():
    def __init__(self):
        self.shops = None
        self.item_categories = None
        self.items = None

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
        return vectors


    def loadInfo(self):
        self.shops = self.loadData('data/shops.csv')
        self.shops = [s.rsplit(',',1)[0][1:-1].replace('\"\"','\"') for s in self.shops[1:]]
        vectors = self.transfer2Vec(self.shops, 0)
        for i in range(len(self.shops)): self.shops[i] = [self.shops[i], vectors[i]]
            
        self.item_categories = self.loadData('data/item_categories.csv')
        self.item_categories = [s.rsplit(',',1)[0] for s in self.item_categories[1:]]
        vectors = self.transfer2Vec(self.item_categories, 0)
        for i in range(len(self.item_categories)): self.item_categories[i] = [self.item_categories[i], vectors[i]]

        self.items = self.loadData('data/items.csv')
        self.items = [[s.rsplit(',',2)[0].replace('\"',''), int(s.rsplit(',',2)[2])] for s in self.items[1:]]
        vectors = self.transfer2Vec([i[0] for i in self.items], 75)
        for i in range(len(self.items)): self.items[i] += [vectors[i],0.0]


    def loadTrainData(self):
        trainX = []
        trainY = []
        rawData = self.loadData('data/sales_train.csv.gz')
        if os.path.exists('trainDataFeatures.npy') and os.path.exists('trainDataLabel.npy'):
            trainX = np.load('trainDataFeatures.npy')
            trainY = np.load('trainDataLabel.npy')
        else:
            for data in rawData[1:]:
                features = []
                units = data.split(',')
                date = [int(u) for u in units[0].split('.')]
                weekday = datetime.datetime(date[2],date[1],date[0]).weekday()
                features += [float(u) for u in units[0].split('.')]
                features.append(float(weekday))
                shop_id = int(units[2])
                features += self.shops[shop_id][1]
                item_id = int(units[3])
                category_id = self.items[item_id][1]
                features += self.items[item_id][2]
                features += self.item_categories[category_id][1]
                item_price = float(units[4])
                self.items[item_id][3] = item_price
                item_cnt_day = float(units[5])
                features.append(item_price)
                trainX.append(np.array(features))
                trainY.append(item_cnt_day)
            trainX = np.array(trainX)
            trainY = np.array(trainY)
            np.save('trainDataFeatures.npy', trainX)
            np.save('trainDataLabel.npy', trainY)
        print(np.shape(trainX)[:2])
        print(np.shape(trainY)[:2])
        return trainX, trainY
            
    def loadTestData(self):
        testX = []
        if os.path.exists('testDataFeatures.npy'):
            testX = np.load('testDataFeatures.npy')
        else:
            rawData = self.loadData('data/test.csv.gz')
            month = 11
            year = 2015
            for data in rawData[1:]:
                units = data.split(',')
                shop_id = int(units[1])
                item_id = int(units[2])
                category_id = self.items[item_id][1]
                for day in range(1,31):
                    features = []
                    weekday = datetime.datetime(year, month, day).weekday()
                    features += [float(day), float(month), float(year), float(weekday)]
                    features += self.shops[shop_id][1]
                    features += self.items[item_id][2]
                    features += self.item_categories[category_id][1]
                    features.append(self.items[item_id][3])
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