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



def meanencoding_lagfeature():
	print("[%s] Initialize the dataset." % logging.time.ctime())
	dataset = DataSet()
	print("[%s] Loading Training Data ..." % logging.time.ctime())
	trainX, trainY = dataset.loadTrainData(True)
	print("[%s] Loading Testing Data ..." % logging.time.ctime())
	testX = dataset.loadTestData(True)

	train_df = pd.DataFrame(trainX,columns = ['shop_id', 'item_id', 'cat_id', 'date_block_num','year', 'month',  'city_code', 'type_id','sub_type_id','price'])
	train_df = train_df.drop(['price'], axis=1)
	train_df['item_cnt_month'] = np.array(trainY)
	test_df = pd.DataFrame(testX,columns = ['shop_id', 'item_id', 'cat_id','date_block_num', 'year', 'month',  'city_code', 'type_id','sub_type_id', 'price'])
	test_df = test_df.drop(['price'], axis=1)
	train_row = int(train_df.shape[0])
	print(train_row)
	test_count = len(test_df)



	print("[%s] Mean Encoding and Feature Engineering ..." % logging.time.ctime())
	# K fold Target Encoding
	print('%0.2f min: Start adding mean-encoding for item_cnt_month'%((time.time() - start_time)/60))
	Target = 'item_cnt_month'
	global_mean = train_df[Target].mean()

	SEED = 0
	kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)

	mean_encoded_columns = ['shop_id', 'item_id', 'cat_id','city_code', 'type_id','sub_type_id']
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
		all_test_index = np.arange(test_count)
		temp = test_df.iloc[all_test_index]
		test_df[added_column_name] = np.nan
		test_df.loc[:,added_column_name] = \
			temp[column].map(train_df.groupby(column)[Target].mean())
		#print(test_df[test_df[[added_column_name]].isnull().any(axis=1)])
		
	test_df.fillna(0, inplace = True)
	print('%0.2f min: Finish adding mean-encoding'%((time.time() - start_time)/60))


	# Feature Engineering -- Creating lag based feature 
	print('%0.2f min: Start adding lag based feature'%((time.time() - start_time)/60))
	# add one column to concat
	test_df['item_cnt_month'] = -1
	train_test_df = pd.concat([train_df, test_df], axis = 0)

	print('%0.2f min: Adding first lag feature -- x:1,2,3,6,12 month ago item_cnt_month with same shop_id&item_id'%((time.time() - start_time)/60))
	lookback_range = [1,2,3,6,12]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_same_shop_item'
		train_test_df_temp = train_test_df.copy()
		train_test_df_temp.loc[:,'date_block_num'] += diff
		train_test_df_temp.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(train_test_df_temp[['shop_id_cnt_month_mean_Kfold','item_id_cnt_month_mean_Kfold','date_block_num', new_feature_name]],\
		 		on = ['shop_id_cnt_month_mean_Kfold','item_id_cnt_month_mean_Kfold','date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)

	print('%0.2f min: Adding second lag feature -- x:1 month ago average item_cnt_month in all'%((time.time() - start_time)/60))
	groups = train_test_df.groupby(by = ['date_block_num'])
	lookback_range = [1]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_in_all'
		result = groups.agg({'item_cnt_month':'mean'})
		result = result.reset_index()
		result.loc[:,'date_block_num'] += diff
		result.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(result, on = ['date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)
	
	print('%0.2f min: Adding third lag feature -- x:1,2,3,6,12 month ago average item_cnt_month with same item_id'%((time.time() - start_time)/60))
	groups = train_test_df.groupby(by = ['item_id_cnt_month_mean_Kfold','date_block_num'])
	lookback_range = [1,2,3,6,12]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_item'
		result = groups.agg({'item_cnt_month':'mean'})
		result = result.reset_index()
		result.loc[:,'date_block_num'] += diff
		result.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(result, on = ['item_id_cnt_month_mean_Kfold','date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)
	
	print('%0.2f min: Adding fourth lag feature -- x:1,2,3,6,12 month ago average item_cnt_month with same shop_id'%((time.time() - start_time)/60))
	groups = train_test_df.groupby(by = ['shop_id_cnt_month_mean_Kfold','date_block_num'])
	lookback_range = [1,2,3,6,12]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_shop'
		result = groups.agg({'item_cnt_month':'mean'})
		result = result.reset_index()
		result.loc[:,'date_block_num'] += diff
		result.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(result, on = ['shop_id_cnt_month_mean_Kfold','date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)
	
	print('%0.2f min: Adding fifth lag feature -- x:1 month ago average item_cnt_month with same cat_id'%((time.time() - start_time)/60))
	groups = train_test_df.groupby(by = ['cat_id_cnt_month_mean_Kfold','date_block_num'])
	lookback_range = [1]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_cat'
		result = groups.agg({'item_cnt_month':'mean'})
		result = result.reset_index()
		result.loc[:,'date_block_num'] += diff
		result.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(result, on = ['cat_id_cnt_month_mean_Kfold','date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)

	print('%0.2f min: Adding sixth lag feature -- x:1 month ago average item_cnt_month with same cat_id&shop_id'%((time.time() - start_time)/60))
	groups = train_test_df.groupby(by = ['cat_id_cnt_month_mean_Kfold','shop_id_cnt_month_mean_Kfold','date_block_num'])
	lookback_range = [1]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_cat_shop'
		result = groups.agg({'item_cnt_month':'mean'})
		result = result.reset_index()
		result.loc[:,'date_block_num'] += diff
		result.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(result, on = ['cat_id_cnt_month_mean_Kfold','shop_id_cnt_month_mean_Kfold','date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)

	print('%0.2f min: Adding seventh lag feature -- x:1 month ago average item_cnt_month with same city_code'%((time.time() - start_time)/60))
	groups = train_test_df.groupby(by = ['city_code_cnt_month_mean_Kfold','date_block_num'])
	lookback_range = [1]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_city'
		result = groups.agg({'item_cnt_month':'mean'})
		result = result.reset_index()
		result.loc[:,'date_block_num'] += diff
		result.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(result, on = ['city_code_cnt_month_mean_Kfold','date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)

	print('%0.2f min: Adding eighth lag feature -- x:1 month ago average item_cnt_month with same city_code&shop_id'%((time.time() - start_time)/60))
	groups = train_test_df.groupby(by = ['shop_id_cnt_month_mean_Kfold','city_code_cnt_month_mean_Kfold','date_block_num'])
	lookback_range = [1]
	for diff in tqdm(lookback_range):
		new_feature_name = str(diff) + '_month_ago_item_cnt_month_shop_city'
		result = groups.agg({'item_cnt_month':'mean'})
		result = result.reset_index()
		result.loc[:,'date_block_num'] += diff
		result.rename(columns={'item_cnt_month':new_feature_name}, inplace = True)
		train_test_df = train_test_df.merge(result, on = ['shop_id_cnt_month_mean_Kfold','city_code_cnt_month_mean_Kfold','date_block_num'], how = 'left')
		train_test_df[new_feature_name] = train_test_df[new_feature_name].fillna(0)

	print('%0.2f min: Finish adding lag based feature'%((time.time() - start_time)/60))


	print('%0.2f min: Start generating training and testing data'%((time.time() - start_time)/60))
	#split new train_df and test_df from train_test_df
	new_train_df = train_test_df.iloc[:train_row]
	new_test_df = train_test_df.iloc[train_row:]

	new_train_df = new_train_df.drop(['shop_id','item_id','cat_id','date_block_num','item_cnt_month','shop_id_cnt_month_mean_Kfold', 'item_id_cnt_month_mean_Kfold', 'cat_id_cnt_month_mean_Kfold'], axis=1)
	new_test_df = new_test_df.drop(['shop_id','item_id','cat_id','date_block_num','item_cnt_month','shop_id_cnt_month_mean_Kfold', 'item_id_cnt_month_mean_Kfold', 'cat_id_cnt_month_mean_Kfold'], axis=1)

	new_train_df.head().to_csv('train_df_head.csv')
	new_test_df.head().to_csv('test_df_head.csv')
	mean_encoded_lag_feature_trainX = np.array(new_train_df.values.tolist())
	mean_encoded_lag_feature_testX = np.array(new_test_df.values.tolist())
	
	print(np.shape(mean_encoded_lag_feature_trainX))
	print(np.shape(mean_encoded_lag_feature_testX))
	print('%0.2f min: Finish generating training and testing data'%((time.time() - start_time)/60))


	return mean_encoded_lag_feature_trainX, trainY, mean_encoded_lag_feature_testX


if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    mean_encoded_trainX, trainY, mean_encoded_testX = meanencoding_lagfeature()