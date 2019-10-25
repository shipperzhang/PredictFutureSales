import os
import re
import gzip
import numpy as np
from googletrans import Translator
from gensim.models import Word2Vec

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

    def transfer2Vec(self, words_ru):
        translator = Translator()
        words_en = translator.translate(' | '.join(words_ru)).text.split('|')
        sentence = [[s.strip()] + re.sub(r'[,\"\(\)]','',s.strip()).split() + [s.strip()] for s in words_en]
        window = max([len(s) - 2 for s in sentence])
        model = Word2Vec(sentence, min_count=2, size=50, window=window)
        res = []
        for word in words_en:
            res.append(model.wv[word.strip()])
        return res


    def loadInfo(self):
        self.shops = self.loadData('data/shops.csv')
        self.shops = [s.rsplit(',',1)[0][1:-1].replace('\"\"','\"') for s in self.shops[1:]]
        vectors = self.transfer2Vec(self.shops)
        for i in range(len(self.shops)): self.shops[i] = [self.shops[i], vectors[i]]
            
        self.item_categories = self.loadData('data/item_categories.csv')
        self.item_categories = [s.rsplit(',',1)[0] for s in self.item_categories[1:]]
        vectors = self.transfer2Vec(self.item_categories)
        for i in range(len(self.item_categories)): self.item_categories[i] = [self.item_categories[i], vectors[i]]

        # items = loadData('data/items.csv')
        # items = [(s.rsplit(',',2)[0].replace('\"',''), s.rsplit(',',2)[2]) for s in items[1:]]


    def loadTrainData(self):
        rawData = self.loadData('data/sales_train.csv.gz')
        trainX = []
        trainY = []
        for data in rawData[1:]:
            features = []
            units = data.split(',')
            # date = units[0]
            # shop_id = int(units[2])
            # item_id = int(units[3])
            item_price = float(units[4])
            item_cnt_day = float(units[5])
            features.append(item_price)
            trainX.append(features)
            trainY.append(item_cnt_day)
        trainX = np.array(trainX)
        trainY = np.array(trainY)
        return trainX, trainY
            
    def loadTestData(self):
        pass

if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    dataset = DataSet()
    dataset.loadInfo()