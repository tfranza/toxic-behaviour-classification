
import pandas as pd
import requests

BADWORDS_LINK = 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'    # common toxic badwords

class FeaturesExtractor():
	"""
	Class for dealing with the extraction of engineered features from the input text.

	Args:
	::param text; text to be processed
	"""
	def __init__(
		self, 
		text: str
	):
		super().__init__(self)
		self.text = text
		self.features = text.iloc[:,0:0]						


	"""
	Method that applies feature engineering on the loaded data. It may involve several operations 
	that are applied in the order as requested in the features list.

	::param features_list; list of features to extract from the text
	"""
	def extract(
		self, 
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

	'''
	Method to extract word counts that involve explicit offense or that have toxicity in them. 
	'''
	def extract_badwords(self):
		html_doc = requests.get(BADWORDS_LINK)
		badwords = html_doc.text.split('\n')[1:-1]
		return self.text.apply(lambda x: sum(1 for word in x.split(' ') if word in badwords))

	'''
	Method that extracts the sentence length as a measure of the number of letters.
	'''
	def extract_sentence_length(self):
		return self.text.apply(len)

	'''
	Method that extracts the amount of exclamation marks in the sentence.
	'''
	def extract_exclamation_marks(self):
		return self.text.apply(lambda x: len(x.split('!'))-1)

	'''
	Method that extracts the amount of interrogation marks in the sentence.
	'''
	def extract_interrogation_marks(self):
		return self.text.apply(lambda x: len(x.split('?'))-1)

	'''
	Method that extracts the amount of upper words in the sentence.
	'''
	def extract_upper_words(self):
		return self.text.apply(lambda x: sum(1 for word in x.split(' ') if word.isupper()))

	'''
	Method that extracts the amount of upper letters in the sentence.
	'''
	def extract_upper_letters(self):
		return self.text.apply(lambda x: sum(1 for letter in x if letter.isupper()))

