import os
import gzip
import csv
from dataset import DataSet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import numpy as np
import logging
import pandas as pd
import time
start_time = time.time()
from tqdm import tqdm
from sklearn.model_selection import KFold
import dask.dataframe as dd


def meanencoding():
	print("[%s] Initialize the dataset." % logging.time.ctime())
	dataset = DataSet()
	print("[%s] Loading Training Data ..." % logging.time.ctime())
	trainX, trainY = dataset.loadTrainData(True)
	print("[%s] Loading Testing Data ..." % logging.time.ctime())
	testX = dataset.loadTestData(True)

	train_df = pd.DataFrame(trainX,columns = ['shop_id', 'item_id', 'cat_id', 'year', 'month',  'price'])
	train_df = train_df.drop(['price'], axis=1)
	train_df['item_cnt_month'] = np.array(trainY)
	test_df = pd.DataFrame(testX,columns = ['shop_id', 'item_id', 'cat_id', 'year', 'month',  'price'])
	test_df = test_df.drop(['price'], axis=1)
	test_count = len(test_df)


	# K fold Target Encoding
	print('%0.2f min: Start adding mean-encoding for item_cnt_month'%((time.time() - start_time)/60))
	Target = 'item_cnt_month'
	global_mean = train_df[Target].mean()

	SEED = 0
	kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)

	mean_encoded_columns = ['shop_id', 'item_id', 'cat_id']
	for column in tqdm(mean_encoded_columns):
		added_column_name = column + '_cnt_month_mean_Kfold'
		df_temp = train_df[[column]+[Target]]
		df_temp[added_column_name] = global_mean
		for tr_ind, val_ind in kf.split(df_temp):
			X_tr, X_val = df_temp.iloc[tr_ind], df_temp.iloc[val_ind]
			df_temp.loc[df_temp.index[val_ind], added_column_name] = \
						X_val[column].map(X_tr.groupby(column)[Target].mean())
		df_temp[added_column_name].fillna(global_mean, inplace = True)


		train_df = pd.concat([train_df, df_temp[added_column_name]],axis = 1) 
		
	# Adding target mean encoding for test DF
	train_temp = train_df.copy()
	mean_encoded_columns = ['shop_id', 'item_id', 'cat_id']
	for column in tqdm(mean_encoded_columns):
		temp = pd.DataFrame(np.nan, index = np.arange(test_count), columns = [column])
		added_column_name = column + '_cnt_month_mean_Kfold'
		test_df[added_column_name] = np.nan
		test_df.loc[:,  [added_column_name]] = \
			temp[column].map(train_temp.groupby(column)[Target].mean())

	train_df = train_df.drop(['item_cnt_month'], axis=1)

	print(train_df.shape)
	print(train_df.head())
	print(test_df.shape)
	print(test_df.head())
	test_df.head(100).to_csv('myfile_test.csv')

	print('%0.2f min: Finish adding mean-encoding'%((time.time() - start_time)/60))

	mean_encoded_trainX = np.array(train_df.values.tolist())
	mean_encoded_testX = np.array(test_df.values.tolist())
	# print(mean_encoded_trainX[0],mean_encoded_testX[0])
	print(np.shape(mean_encoded_trainX))
	print(np.shape(mean_encoded_testX))

	return mean_encoded_trainX, trainY, mean_encoded_testX


if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    mean_encoded_trainX, trainY, mean_encoded_testX = meanencoding()