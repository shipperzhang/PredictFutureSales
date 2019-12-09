import os
import gzip
import csv
from dataset import DataSet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import numpy as np
import logging
from meanencoding_lagfeature import meanencoding_lagfeature

def param_tuning(X,Y):
	p_grid = {"max_depth": [i for i in [3,5,7,9]],\
				"min_child_weight":[i for i in [1,3,5]]}


	kf= KFold(n_splits=5, random_state=None, shuffle=False)
	#outer loop start
	for train_index, test_index in kf.split(X):
		#print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
		y_train, y_test = np.array(Y)[train_index], np.array(Y)[test_index]
		# Choose cross-validation techniques for the inner loops
		inner_cv = KFold(n_splits=5, shuffle=True)
		model = xgb.XGBRegressor(max_depth = 6, min_child_weight=3, \
		eta = 0.05, num_round = 200)
		#inner loop
		model = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,iid=False, scoring='neg_mean_squared_error')
		model.fit(X_train, y_train)
		best_params = model.best_params_
		# outer loop calculate accuracy
		mse = model.score(X_test, y_test)
		rmse = np.sqrt(mse)
		print("The parameters chosen by inner CV loop are:  %s, the corresponding testing accuracy is %s\n" % (best_params, accuracy4) )


if __name__ == "__main__":
	mean_encoded_trainX, training_Y, mean_encoded_testX = meanencoding_lagfeature()
	print("[%s] Tune Parameters ..." % logging.time.ctime())
	param_tuning(mean_encoded_trainX,training_Y)

