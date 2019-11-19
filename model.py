import os
import gzip
import csv
from dataset import DataSet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import numpy as np
import logging
from meanencoding import meanencoding

def XGBRegressor_model(X,Y):
	# model
	print('Setting Hyper Parameter ...')
	model = xgb.XGBRegressor(max_depth = 11, min_child_weight=0.5, \
		subsample = 1, eta = 0.3, num_round = 1000, seed = 1)
	print('Train the model ...')
	model = model.fit(X,Y)
	# cross validation using mse
	# print('Crossing Validation ...')
	# cv = KFold(n_splits=5, shuffle=True)
	# mse_scores = cross_val_score(model, X, Y, cv=cv, scoring = 'neg_mean_squared_error')
	# # calculate root mean square error
	# rmse_scores = np.sqrt(mse_scores)
	# # calculate average rmse
	# score = rmse_scores.mean()
	return model

def predict_and_generate_submission_file(model, X):
	cnt_months = model.predict(X)
	n = len(cnt_months)
	csvfile = open('submission.csv', 'w')
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['ID', 'item_cnt_month'])
	for i in range(n):
		if cnt_months[i] >= 20.0: writer.writerow([str(i), '20.0'])
		else: writer.writerow([str(i), str(cnt_months[i])])
	csvfile.close()
	if os.path.exists('submission.csv.gz'):
		os.system('rm submission.csv.gz')
	os.system('gzip submission.csv')

if __name__ == "__main__":
	#print("[%s] Initialize the dataset." % logging.time.ctime())
	#dataset = DataSet()
	#print("[%s] Loading Training Data ..." % logging.time.ctime())
	#training_X, training_Y = dataset.loadTrainData(True)
	#print("[%s] Loading Testing Data ..." % logging.time.ctime())
	#testing_X = dataset.loadTestData(True)
	mean_encoded_trainX, training_Y, mean_encoded_testX = meanencoding()
	print("[%s] Train Model ..." % logging.time.ctime())
	model = XGBRegressor_model(mean_encoded_trainX, training_Y)
	print("[%s] Predicting ..." % logging.time.ctime())
	predict_and_generate_submission_file(model, mean_encoded_testX)
	print("[%s] Done." % logging.time.ctime())
