from dataset import DataSet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import numpy as np

def XGBRegressor_model(X,y):
	# model
	model = xgb.XGBRegressor(max_depth = 10, min_child_weight=0.5, \
		subsample = 1, eta = 0.3, num_round = 1000, seed = 1)
	# cross validation using mse
	cv = KFold(n_splits=5, shuffle=True)
	mse_scores = cross_val_score(model, X, y, cv=cv, scoring = 'neg_mean_squared_error')
	# calculate root mean square error
	rmse_scores = np.sqrt(mse_scores)
	# calculate average rmse
	score = rmse_scores.mean()
	return model, score

def predict_and_generate_submission_file(model, X):
	# y_pred = model.predict(X)
	pass


if __name__ == "__main__":
    """
    For Test And Debug Only
    """
    dataset = DataSet()
    training_X, training_y = dataset.loadTrainData()
    # testing_X = dataset.loadTestData()