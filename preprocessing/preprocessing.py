
import pandas as pd
import requests

from ./engineering import FeaturesExtractor
from ./cleaning import TextCleaner

class Preprocessor():
	"""
	Class for dealing with the general preprocessing of the data. It stores preprocessing methods 
	that can recall either the cleaning or the feature engineering process. 

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
		labels = self.df.iloc[:,1:]								
		self.labels = labels if labels.shape[1]!=0 else None
		
		self.X = df.iloc[:,[0]].values
		self.y = df.iloc[:,1:].values

		self.features = self.df.iloc[:,0:0]

	######################################################################################
    ######### PREPROCESSING METHODS

	"""
	Method to apply text cleaning on the loaded data. 
	"""
	def apply_text_cleaning(
		self, 
		operations_list: list = []
	):
		self.text = TextCleaner(self.text).brush(operations_list)
		return self

	"""
	Method to apply feature engineering on the loaded data. 
	"""
	def apply_feature_engineering(
		self, 
		features_list: list = []
	):
		self.features = FeatureExtractor(self.text).extract(features_list)
		return self

    ######################################################################################
    ######### GETTER METHODS

	"""
	Getter method for the variable containing the entire dataframe.
	"""
	@property
	def data(self):
		return self.df if self.df else None

	"""
	Getter method for the variable containing the textual data.
	"""
	@property
	def text(self):
		return self.text if self.text else None

	"""
	Getter method for the variable containing the features engineered from the data.
	"""
	@property
	def features(self):
		return self.features if self.features else None

	"""
	Getter for the variable containing the data labels.
	"""
	@property
	def labels(self):
		return self.labels if self.labels else None


