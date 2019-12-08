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
        self.shops = []
        self.item_categories = []
        self.items = []
        self.general_prices = []
        self.cloest_prices = []

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


    def loadInfo(self):
        """
        Load basic information for shops, items & item_categories
        """
        cities = ['Yakutsk', 'Adygea', 'Balashikha', 'Volzhsky', 'Vologda', 'Voronezh', 
                'Outbound Trade', 'Zhukovsky', 'Online', 'Kazan', 'Kaluga', 
                'Kolomna', 'Krasnoyarsk', 'Kursk', 'Moscow', 'Mytishchi', 'Nizhny Novgorod', 
                'Novosibirsk', 'Omsk', 'Rostov-on-Don','St. Petersburg','Samara','Sergiev Posad',
                'Surgut','Tomsk','Tyumen','Ufa','Khimki','Chekhov','Yaroslavl']
        self.shops = self.loadDataFromFile('data/shops-translated.csv')
        self.shops = [s.split(',',1)[1] for s in self.shops[1:]]
        self.shops = [re.sub(r'[\"\(\)]',' ', s) for s in self.shops]
        for i in range(len(self.shops)):
            for j in range(len(cities)):
                if cities[j] in self.shops[i]:
                    self.shops[i] = j
                    break
        
        cat_type = ['Accessories','Tickets','Delivery','Game Consoles',
                    'Android games','MAC Games','PC Games','Payment cards','Cinema',
                    'Books','Music','Gifts','Programs','Service','Clean media','Batteries','PC','Games']
        cat_subtype = {'Accessories':['PS2','PS3','PS4','PSP','PSVita','XBOX 360','XBOX ONE'],
                       'Game Consoles':['PS2','PS3','PS4','PSP','PSVita','XBOX 360','XBOX ONE','Other'],
                       'Games':['PS2','PS3','PS4','PSP','PSVita','XBOX 360','XBOX ONE','Accessories for games'],
                       'PC Games':['Digit','Additional Edition', 'Collectors Edition', 'Standard Edition'],
                       'Payment cards':['Movies, Music, Games','Live!','PSN','Windows (Digital)'],
                       'Cinema':['Blu-Ray','DVD','Collectors'],
                       'Books':['Artbook, encyclopedia','Audiobooks', 'Business Literature','Comics, manga','Computer Literature','Methodical materials 1C','Postcards','Cognitive Literature','Travel guides','Fiction','Number'],
                       'Music':['CD of local production','CD of branded production','MP3','Vinyl','Musical video','Gift edition'],
                       'Gifts':['Attributes','Gadgets, robots, sports','Soft toys','Board Games','Postcards, stickers','Development','Certificates, services','Souvenirs','Bags, Albums, Mouse pads','Figures'],
                       'Programs':['1C: Enterprise 8','MAC (Number)','Home and Office','Teaching'],
                       'Service':[' ','Tickets'],'Clean media':['spire','piece']}    
        self.item_categories = self.loadDataFromFile('data/item_categories-translated.csv')
        self.item_categories = [s.strip().split(',',1)[1].split(' - ') for s in self.item_categories[1:]]
        # print(self.item_categories)
        for i in range(len(self.item_categories)):
            for j in range(len(cat_type)):
                if cat_type[j] == self.item_categories[i][0]:
                    self.item_categories[i][0] = j
                    break
            if len(self.item_categories[i]) == 1: 
                self.item_categories[i].append(0)
                continue
            subtypes = cat_subtype.get(cat_type[self.item_categories[i][0]],[])
            if not subtypes:
                self.item_categories[i][1] = 0
                continue
            for j in range(len(subtypes)):
                if subtypes[j] in self.item_categories[i][1]:
                    self.item_categories[i][1] = j
                    break
        
        self.items = self.loadDataFromFile('data/items.csv')
        self.items = [int(s.rsplit(',',2)[2]) for s in self.items[1:]]
        self.general_prices = [None] * len(self.items)
        self.cloest_prices = [None] * len(self.items)

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
            for data in testData[1:]:
                shop_id = int(data.split(',')[1])
                item_id = int(data.split(',')[2])
                key = str(shop_id) + ',' + str(item_id)
                pairsInTest.add(key)

            train = pd.read_csv('./data/sales_train.csv.gz')
            train = train.loc[train['item_cnt_day'] >= -1.0].loc[train['item_cnt_day'] <= 20.0].loc[train['item_price'] <= 2000.0].loc[train['item_price'] >= 0]
            train = train.groupby(["date_block_num","shop_id", "item_id"])
            train = train.aggregate({"item_price":np.mean, "item_cnt_day":np.sum}).fillna(0)
            train.reset_index(level=["date_block_num", "shop_id", "item_id"], inplace=True)
            train['item_cnt_day'] = train['item_cnt_day'].clip(0,20)

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
                date_blocks[block_num][key] = [item_price, item_cnt_month]
            
            # generate features
            shopsInTrain = set()
            itemsInTrain = set()
            for i in range(34):
                month = 1 + ( i % 12 )
                year = 2013 + ( i // 12)
                pairsInTrain = set()
                this_month = [None] * len(self.items)
                for key in date_blocks[i].keys():
                    features = []
                    shop_id = int(key.split(',')[0])
                    item_id = int(key.split(',')[1])
                    category_id = self.items[item_id]
                    item_price = date_blocks[i][key][0]
                    item_cnt_month = date_blocks[i][key][1]

                    features += [shop_id, item_id, category_id]
                    features.append(i)
                    features += [year, month]
                    features.append(self.shops[shop_id])
                    features.append(self.item_categories[category_id][0])
                    features.append(self.item_categories[category_id][1])
                    features.append(item_price)
                    
                    if this_month[item_id] == None:
                        this_month[item_id] = [item_price, 1]
                    else:
                        ave, count = this_month[item_id]
                        ave = (ave * count + item_price) / count
                        this_month[item_id] = [ave, count + 1]

                    pairsInTrain.add(key)
                    shopsInTrain.add(shop_id)
                    itemsInTrain.add(item_id)
                    trainX.append(np.array(features))
                    trainY.append(item_cnt_month)

                pairsNotInTrain = set()
                for key in pairsInTest.difference(pairsInTrain):
                    shop_id = int(key.split(',')[0])
                    item_id = int(key.split(',')[1])
                    if shop_id in shopsInTrain and item_id in itemsInTrain: 
                        pairsNotInTrain.add(key)

                for key in pairsNotInTrain:
                    features = []
                    shop_id = int(key.split(',')[0])
                    item_id = int(key.split(',')[1])
                    category_id = self.items[item_id]
                    features += [shop_id, item_id, category_id]
                    features.append(i)
                    features += [year, month]
                    features.append(self.shops[shop_id])
                    features.append(self.item_categories[category_id][0])
                    features.append(self.item_categories[category_id][1])

                    if self.cloest_prices[item_id] == None:
                        print(i,item_id)
                        features.append(this_month[item_id][0])
                    else:
                        previous = self.cloest_prices[item_id]
                        general = self.general_prices[item_id][0]
                        item_price = previous * (1 + ((previous - general) / general))
                        features.append(item_price)

                    trainX.append(np.array(features))
                    trainY.append(0.0)
                
                for i in range(len(this_month)):
                    if this_month[i] == None: continue
                    ave, count = this_month[i]
                    self.cloest_prices[i] = ave
                    if self.general_prices[i] == None: 
                        self.general_prices[i] = [ave, count]
                    else:
                        gave, gcount = self.general_prices[i]
                        gave = (gave * gcount + ave * count) / (gcount + count)
                        self.general_prices[i] = [gave, gcount + count]
            
            trainX = np.array(trainX)
            trainY = np.array(trainY)

            # Save the data for the future convenience
            # np.save('trainDataFeatures.npy', trainX)
            # np.save('trainDataLabel.npy', trainY)

        print(np.shape(trainX)[:2])
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
                category_id = self.items[item_id]
                features += [shop_id, item_id, category_id]
                features.append(34)
                features += [year, month]
                features.append(self.shops[shop_id])
                features.append(self.item_categories[category_id][0])
                features.append(self.item_categories[category_id][1])

                previous = self.cloest_prices[item_id]
                general = self.general_prices[item_id][0]
                item_price = previous * (1 + ((previous - general) / general))
                features.append(item_price)
                
                testX.append(np.array(features))
            testX = np.array(testX)

            # Save the data for the future convenience
            # np.save('testDataFeatures.npy', testX)
        print(np.shape(testX)[:2])
        return testX

if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    dataset = DataSet()
    # dataset.loadInfo()
    trainX, trainY = dataset.loadTrainData(True)
    testX = dataset.loadTestData(True)
    print(testX[:100])
    # print(trainX[0])
    # print('\n')
    # print(testX[0])

