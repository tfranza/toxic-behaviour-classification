
import pandas as pd
import requests

from ./engineering import FeaturesExtractor

class Preprocessor():
	"""
	Class for dealing with the cleaning of the data, whether it is the training set or the test set. 
	It stores preprocessing functions to be applied to the features or labels separately.

	Args:
	::param filename; filename of the data to be processed
	::param path; path where the filename can be found
	"""
	def __init__(
		self, 
		filename: str = '', 
		pathfile: str = 'data/'
	):
		super().__init__(self)
		self.df = pd.read_csv(pathfile + filename, index_col=0)		

		self.text = self.df.iloc[:,[0]]						
		self.features = self.df.iloc[:,0:0]
		labels = self.df.iloc[:,1:]								
		self.labels = labels if labels.shape[1]!=0 else None

	"""
	Function to apply feature engineering on the loaded data. It involves 
	"""
	def apply_feature_engineering(
		self, 
		features_list: list = []
	):
		self.features = FeatureExtractor(self.text).extract(features_list)
		return self

	@property
	def data(self):
		return self.df if self.df else None

	@property
	def text(self):
		return self.text if self.text else None

	@property
	def features(self):
		return self.features if self.features else None

	@property
	def labels(self):
		return self.labels if self.labels else None


