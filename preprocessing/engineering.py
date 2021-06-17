
import numpy as np
import requests

from sklearn.feature_extraction.text import CountVectorizer

BADWORDS_LINK = 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'    # common toxic badwords
MAX_COUNT_FEATURES = 10000

class FeaturesExtractor():
	"""
	Class for dealing with the extraction of engineered features from the input text.

	Args:
	::param text; text to be processed
	::param test; indicates whether we're working on the test set or not 
	"""
	def __init__(
		self, 
		text: str,
		test: bool = False,
	):
		super().__init__(self)
		self.text = text.values
		self.test = test
		self.features = np.zeros(shape=(self.text.shape[0],0))

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
			if feature_name == 'n_badwords':
				new_features = self.extract_n_badwords()
			elif feature == 'sent_len':
				new_features = self.extract_sentence_length()
			elif feature == 'n_!':
				new_features = self.extract_n_exclamation_marks()
			elif feature == 'n_?':
				new_features = self.extract_n_interrogation_marks()
			elif feature == 'n_upper_words':
				new_features = self.extract_n_upper_words()
			elif feature == 'n_upper_letters':
				new_features = self.extract_n_upper_letters()
			elif feature == 'word_counts':
				new_features = self.extract_word_counts_tfidf('counts')
			elif feature == 'word_tfidf':
				new_features = self.extract_word_counts_tfidf('tfidf')
			else:
				log.info(f' - {feature} not found as an option available for selection.')
				continue
			self.features = np.append(self.features, new_features) 

		log.info(' > FEATURE ENGINEERING > completed')
		return self

	'''
	Method to extract word counts that involve explicit offense or that have toxicity in them. 
	'''
	def extract_n_badwords(self):
		html_doc = requests.get(BADWORDS_LINK)
		badwords = html_doc.text.split('\n')[1:-1]
		f = lambda x: sum(1 for word in x.split(' ') if word in badwords)
		return np.array([f(x) for x in self.text])

	'''
	Method that extracts the sentence length as a measure of the number of letters.
	'''
	def extract_sentence_length(self):
		f = lambda x: len(x)
		return np.array([f(x) for x in self.text])

	'''
	Method that extracts the amount of exclamation marks in the sentence.
	'''
	def extract_n_exclamation_marks(self):
		f = lambda x: len(x.split('!'))-1
		return np.array([f(x) for x in self.text])

	'''
	Method that extracts the amount of interrogation marks in the sentence.
	'''
	def extract_n_interrogation_marks(self):
		f = lambda x: len(x.split('?'))-1
		return np.array([f(x) for x in self.text])

	'''
	Method that extracts the amount of upper words in the sentence.
	'''
	def extract_n_upper_words(self):
		f = lambda x: sum(1 for word in x.split(' ') if word.isupper())
		return np.array([f(x) for x in self.text])

	'''
	Method that extracts the amount of upper letters in the sentence.
	'''
	def extract_n_upper_letters(self):
		f = lambda x: sum(1 for letter in x if letter.isupper())
		return np.array([f(x) for x in self.text])

	'''
	Method that extracts the word counts for each word in the data. The amount of words is limited by the 
	MAX_COUNT_FEATURES variable. The vectorizer will try to fit only on the training data.
	
	::param feature_type; type of feature to extract from the text. Allowed values: 'counts', 'tfidf'
	'''
	def extract_word_counts_tfidf(
		self, 
		feature_type: str = 'counts'
	):
		vectorizer = None 
		word_features = None
		try:		# vectorizer already generated		
			with open(f'resources/{feature_type}_vectorizer.pickle', 'rb') as handle:
				vectorizer = pickle.load(handle)
			word_features = vectorizer.transform(self.text.values)
		except:		# vectorizer not yet generated
			if self.test:
				log.info(f' - {feature_type} vectorizer not found!')		
			else:
				if feature_type=='counts':
					vectorizer = CountVectorizer(max_features = MAX_COUNT_FEATURES)
				elif feature_type=='tfidf':
					vectorizer = TfidfVectorizer(max_features = MAX_COUNT_FEATURES)
				word_features = vectorizer.fit_transform(self.text.values)
				with open(f'resources/{feature_type}_vectorizer', 'wb') as handle:
					pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
		return word_features




