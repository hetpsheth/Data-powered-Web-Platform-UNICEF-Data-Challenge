"""
Regressor.py contains the ML model used to predict HDI
Created by: Shantanu Mantri on 11/3/2018
"""
from utils import DataReader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from matplotlib import rcParams
import pandas as pd 
import sys

class Regressor:

	def __init__(self, name, load_model=False):
		"""
		This is a constructor used to make it easier to train/test
		data for any chosen scikit learn regressor
		"""
		self.name = name
		self.name.replace(" ", "")

		if load_model:
			self.model = joblib.load("../models/" + self.name + '.joblib') 

	def train(self, model, save=False, make_chart=False):
		"""
		Trains an input model. Makes Calculations, Charts, and Saves
		the model if necessary.

		Parameters
		----------
		model:     SKLearn Model The regression model to use
		save:      Boolean Whether or not the model should be saved
		make_chart Boolean Whether or not to make/save a chart

		Returns
		-------
		float, float, float: The Average CV Mean Squared Error, Mean Absolute Error, and Test MSE 
		"""
		#get/split data
		reader = DataReader()
		df = reader.create_input_data()
		df = self.preprocess(df)
		self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(df)

		parameters = {'n_estimators' : [1, 5, 10, 20, 30], 'max_depth' : [1, 5, 10]}
		rf = RandomForestRegressor()
		self.model = GridSearchCV(rf, parameters, cv=10)
		#train model
		self.model.fit(self.X_train, self.y_train)

		#Feature importance
		importances = self.model.best_estimator_.feature_importances_
		cols = self.X_train.columns 
		for i in range(len(importances)):
			print(cols[i], importances[i])

		if save:
			joblib.dump(self.model.best_estimator_, "../models/" + self.name + "_2017.joblib")

	
		print("------------------------")
		MSEs = cross_val_score(estimator=self.model,
                         X=self.X_train,
                         y=self.y_train,
                         scoring='neg_mean_squared_error',
                         cv=8)

		predicted = self.model.predict(self.X_test)
		print("Average CV Mean Squared Error: ", abs(np.mean(MSEs)))
		print("Testing Mean Absolute Error: ", mean_absolute_error(self.y_test, self.model.predict(self.X_test)))
		print("Testing MSE: ", mean_squared_error(self.y_test, predicted))
		#print(self.model.feature_importances_)
		if make_chart:
			print("Generating Chart...")
			plt.style.use('dark_background')
			fig, ax = plt.subplots(nrows=1, ncols=1)
			ax.set_ylabel('HDI')
			ax.set_xlabel("Municipality Codmun ID")
			ax.set_title(self.name + 'Real vs Predicted')
			green, = ax.plot(np.arange(20), self.y_test[0:100:5], 'g', label='True')
			red, = ax.plot(np.arange(20), predicted[0:100:5], 'r', label='Predicted')
			ax.set_xticks(np.arange(20))
			x_labels = self.X_test.iloc[0:100:5]['codmun'].tolist()
			ax.set_xticklabels([str(int(y)) for y in x_labels], rotation='vertical')
			plt.legend(handles=[green, red], labels=["True", "Predicted"])
			plt.tight_layout()
			fig.savefig(self.name + "_real_v_predicted")
			for x in range(0, 100, 5):
				print(predicted[x], x_labels[int(x/5)])
			print(x_labels, predicted[0:100:5])

		return np.mean(MSEs), mean_absolute_error(self.y_test, self.model.predict(self.X_test)), mean_squared_error(self.y_test, predicted)

	def predict(self, input_data, year):
		"""
		Predicts values using the most recently trained/loaded model
		
		Parameters
		----------
		input_data: Pandas DataFrame The input columns completely cleaned and for the appropriate model
		
		Returns
		-------
		Numpy Float Array: An array of the resulting HDI Predictions
		"""
		#clean input_data appropriately
		X_cols = list(input_data.columns)
		X_cols.remove('hdi')
		X_cols.remove('Country')
		X_cols.remove('codmun')
		X_cols.remove('entropy_2016')
		X_cols.remove('entropy_2017')
		X_cols.remove('popularity_2016')
		X_cols.remove('popularity_2017')
		X_cols.remove('hdi_estimated_2016')
		X_cols.remove('hdi_estimated_2017')

		if year == 2016:
		####2016####
			X_cols.remove('activity_2017_h0-5')
			X_cols.remove('activity_2017_h6-11')
			X_cols.remove('activity_2017_h12-17')
			X_cols.remove('activity_2017_h18-23')

			X_cols.remove('average_inflow_2017')
			X_cols.remove('average_outflow_2017')

		####2017####
		elif year == 2017:
			X_cols.remove('activity_2016_h0-5')
			X_cols.remove('activity_2016_h6-11')
			X_cols.remove('activity_2016_h12-17')
			X_cols.remove('activity_2016_h18-23')

			X_cols.remove('average_inflow_2016')
			X_cols.remove('average_outflow_2016')

		X = df[X_cols]

		return self.model.predict(X)

	def preprocess(self, df):
		"""
		Cleans data by encoding strings
		Parameters
		----------
		df: Pandas DataFrame The data to be cleaned

		Returns
		-------
		Pandas DataFrame The cleaned data
		"""
		#must be all numerical data
		le = LabelEncoder()
		le.fit(df['Country'])
		trans = le.transform(df['Country'])
		df['Country'] = trans
		return df

	def split_data(self, df):
		#split columns
		y = df['hdi']
		X_cols = list(df.columns)
		X_cols.remove('hdi')
		X_cols.remove('Country')
		X_cols.remove('codmun')
		X_cols.remove('entropy_2016')
		X_cols.remove('entropy_2017')
		X_cols.remove('popularity_2016')
		X_cols.remove('popularity_2017')
		X_cols.remove('hdi_estimated_2016')
		X_cols.remove('hdi_estimated_2017')

		###2016 Predictions####
		#X_cols.remove('activity_2016_h0-5')
		#X_cols.remove('activity_2016_h6-11')
		#X_cols.remove('activity_2016_h12-17')
		#X_cols.remove('activity_2016_h18-23')
		#X_cols.remove('average_inflow_2016')
		#X_cols.remove('average_outflow_2016')

		###2017 Predictions####
		#X_cols.remove('activity_2017_h0-5')
		#X_cols.remove('activity_2017_h6-11')
		#X_cols.remove('activity_2017_h12-17')
		#X_cols.remove('activity_2017_h18-23')
		#X_cols.remove('average_inflow_2017')
		#X_cols.remove('average_outflow_2017')
		
		X = df[X_cols]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
		return X_train.astype('float'), X_test.astype('float'), y_train.astype('float'), y_test.astype('float')






if __name__ == "__main__":
	try:
		options = sys.argv[1]
	except:
		print("Please add --train or --test after py Regressor.py")
		options = None

	if options == "--train":
		r = Regressor("Random Forest", load_model=False)
		mod = Regressor("Random Forest", load_model=False)
		cv, ma, mse = r.train(mod, save=False, make_chart=False)
		print(cv, ma, mse)

	elif options == "--test":
		model_name = sys.argv[2] + " " + sys.argv[3] #Random Forest_2017 or Random Forest_2016
		year = int(model_name.split("_")[-1])
		r = Regressor(model_name, load_model=True)
		reader = DataReader()
		df = reader.create_input_data()
		predictions = r.predict(df, year)
		print("Actual || Predicted")
		for i in range(len(predictions)):
			print(df.iloc[i]['hdi'], "||", predictions[i])

######Training Code#########

#cv_error = []
#testing_ma_error = []
#testing_mse = []
#mod = RandomForestRegressor(bootstrap=True, criterion='mae', n_estimators=100)
#mod = RandomForestRegressor()
#r = Regressor("Random Forest_2017", load_model=True)
#importances = r.model.feature_importances_
#reader = DataReader()
#df = reader.create_input_data()
#df = r.preprocess(df)
#df,a,b,c = r.split_data(df)
#cols = df.columns
#predictions = r.predict(df)
#cv, ma, mse = r.train(mod, save=False, make_chart=False)
#cv_error.append(cv)
#testing_ma_error.append(ma)
#testing_mse.append(mse)
#print predictions
#print(cv_error)
#print(testing_ma_error)
#print(testing_mse)
#print(predictions)
#output_df = pd.DataFrame(columns=['codmun', 'country', 'HDI_2016'])
#output_df['codmun'] = df['codmun']
#output_df['country'] = df['Country']
#output_df['HDI_2017'] = predictions
#print(output_df)
#output_df.to_csv("../data/predictions_2017", index=False)

###Testing###


############################

"""
mod = KNeighborsRegressor(n_neighbors=5)
print("==================")
print("KNN")
r = Regressor(mod)
cv, ma, mse = r.train()
cv_error.append(cv)
testing_ma_error.append(ma)
testing_mse.append(mse)

mod = LinearRegression()
print("==================")
print("Linear Regression")
r = Regressor(mod)
cv, ma, mse = r.train()
cv_error.append(cv)
testing_ma_error.append(ma)
testing_mse.append(mse)

mod = svm.SVR()
print("==================")
print("SVM")
r = Regressor(mod)
cv, ma, mse = r.train()
cv_error.append(cv)
testing_ma_error.append(ma)
testing_mse.append(mse)

mod = BayesianRidge(n_iter=300)
print("==================")
print("Bayesian Ridge")
r = Regressor(mod)
cv, ma, mse = r.train()
cv_error.append(cv)
testing_ma_error.append(ma)
testing_mse.append(mse)
"""
###Create a chart###
#fig, ax = plt.subplots()
#idx = np.arange(1, 6)
#mse_fam = [0.019078001287188905, 0.019256912600907165, 0.020155023469181438,
#0.020715330582168442, 0.028051791020443233]
#a, b, c, d, e = plt.bar(idx, mse_fam)
#a.set_facecolor('r')
#b.set_facecolor('g')
#c.set_facecolor('b')
#d.set_facecolor('k')
#e.set_facecolor('c')
#ax.set_xticks(idx)
#ax.set_xticklabels(["All Features", "No Twitter", "No Migration", "No Twitter, Migration", "No OSM Data"], rotation=90)
#ax.set_ylabel('Mean Squared Error')
#ax.set_title('Input Features and Error')
#plt.tight_layout()
#fig.savefig("cv_error")

"""
fig, ax = plt.subplots()
a, b, c, d, e = plt.bar(idx, testing_ma_error)
a.set_facecolor('r')
b.set_facecolor('g')
c.set_facecolor('b')
d.set_facecolor('k')
e.set_facecolor('c')
ax.set_xticks(idx)
ax.set_xticklabels(["Random Forest", "KNN", "Linear Regression", "SVM", "Bayesian Ridge"])
ax.set_ylabel('Testing Mean Absolute Error')
ax.set_title('Testing Mean Absolute Error by Regressor')
fig.savefig("testing_mae")


fig, ax = plt.subplots()
a, b, c, d, e = plt.bar(idx, testing_mse)
a.set_facecolor('r')
b.set_facecolor('g')
c.set_facecolor('b')
d.set_facecolor('k')
e.set_facecolor('c')
ax.set_xticks(idx)
ax.set_xticklabels(["Random Forest", "KNN", "Linear Regression", "SVM", "Bayesian Ridge"])
ax.set_ylabel('Testing Mean Squared Error')
ax.set_title('Testing Mean Squared Error by Regressor')
fig.savefig("testing_mse")
"""

#fig, ax = plt.subplots()
#plt.bar(range(len(importances)), importances)
#ax.set_xticks([i for i in range(len(importances))])
#ax.set_xticklabels(cols, rotation='vertical')
#ax.set_ylabel('Importance')
#ax.set_title('Feature Importances for HDI 2017 Prediction')
#plt.tight_layout()
#fig.savefig("../diagrams/feature_importance_2017")
