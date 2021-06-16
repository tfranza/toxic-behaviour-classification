
import pandas as pd
import requests

BADWORDS_LINK = 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'    # common toxic badwords

class FeaturesExtractor():
	"""
	Class for dealing with the extraction of features from the input text.

	Args:
	::param filename; filename of the data to be processed
	::param path; path where the filename can be found
	"""
	def __init__(
		self, 
		text: str
	):
		super().__init__(self)
		self.text = text
		self.features = text.iloc[:,0:0]						

	######################################################################################
	#####  FEATURE ENGINEERING

	"""
	Function to apply feature engineering on the loaded data. It involves 
	"""
	def apply_feature_engineering(self, 
		features_list: list = []
	):
		if not features_list:
			log.info(' > FEATURE ENGINEERING > aborted: empty features list!')
			return self
		else:
			log.info(' > FEATURE ENGINEERING > started')

		for feature_name in features_list:
			log.info(f' - {feature} extraction...')
			if feature_name == 'badwords':
				self.features[[feature]] = self.extract_badwords()
			elif feature == 'sent_len':
				self.features[[feature]] = self.extract_sentence_length()
			elif feature == '!_len':
				self.features[[feature]] = self.extract_exclamation_marks()
			elif feature == '?_len':
				self.features[[feature]] = self.extract_interrogation_marks()
			elif feature == 'upper_words':
				self.features[[feature]] = self.extract_upper_words()
			elif feature == 'upper_letters':
				self.features[[feature]] = self.extract_upper_letters()

		log.info(' > FEATURE ENGINEERING > completed')
		return self

	def extract_badwords(self):
		html_doc = requests.get(BADWORDS_LINK)
		badwords = html_doc.text.split('\n')[1:-1]
		return self.text.apply(lambda x: sum(1 for word in x.split(' ') if word in badwords))

	def extract_sentence_length(self):
		return self.text.apply(len)

	def extract_exclamation_marks(self):
		return self.text.apply(lambda x: len(x.split('!'))-1)

	def extract_exclamation_marks(self):
		return self.text.apply(lambda x: len(x.split('?'))-1)

	def extract_upper_words(self):
		return self.text.apply(lambda x: sum(1 for word in x.split(' ') if word.isupper()))

	def extract_upper_letters(self):
		return self.text.apply(lambda x: sum(1 for letter in x if letter.isupper()))

