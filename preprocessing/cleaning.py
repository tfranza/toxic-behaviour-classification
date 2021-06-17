
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

STOPWORDS = set(stopwords.words('english'))

ABBREVIATIONS = {
        'can\'t': 'can not',
        'n\'t': ' not',
        '\'m': ' am',
        '\'s': ' is',
        '\'re': ' are',
        '\'d': '  would',
        '\'ve': ' have',
        '\'ll': ' will',
        ' y\'all ': ' you all ',
        '\'cause': ' because',
        '\'bout': ' about',
        '\'em': ' them',
        'g\'day ': ' good day',
        'c\'mon': ' come on',
        'wp': ' well played',
        ' u ': ' you '
    }

PUNCTUATIONS = '.,"`)(:-;][|{}#&%•*/—=\n'

class TextCleaner():
	"""
	Class for dealing with the cleaning of the data: makes available several methods to do it. 

	Args:
	::param text; text to be processed
	"""
	def __init__(
		self, 
		text: str
	):
		super().__init__(self)
		self.original_text = text
		self.updated_text = text						

	"""
	Method that applies the cleaning operations on the loaded data. It may involve several operations 
	that are applied in the order as requested in the operations list.

	::param operations_list; list of clearning operations to apply on text
	"""
	def brush(
		self, 
		operations_list: list = []
	):
		if not operations_list:
			log.info(' > TEXT CLEANING > aborted: empty operations list!')
			return self
		else:
			log.info(' > TEXT CLEANING > started')

		for operation in operations_list:
			log.info(f' - applying {feature}...')
			if operation == 'lowering':
				self.updated_text = self.lowering()
			elif operation == 'abbreviations_removal':
				self.updated_text = self.abbreviations_removal()
			elif operation == 'punctuations_removal':
				self.updated_text = self.punctuations_removal()
			elif operation == 'numbers_removal':
				self.updated_text = self.numbers_removal()
			elif operation == 'stopwords_removal':
				self.updated_text = self.stopwords_removal()
			elif operation == 'lemmatization':
				self.updated_text = self.lemmatization()
			elif operation == 'spaces_removal':
				self.updated_text = self.spaces_removal()

		log.info(' > TEXT CLEANING > completed')
		return self

	######################################################################################
    ######### CLEANING OPERATION METHODS

	'''
	Method to transform every uppercased letter into a lowercased one. 
	'''
	def lowering(self):
		return self.updated_text.map(str.lower)

	'''
	Method to remove abbreviations from the text. 
	'''
	def abbreviations_removal(self):
	    sents_wo_abbrv = self.updated_text.copy()
	    for abbr in ABBREVIATIONS:
	        abbr_conv = abbreviations[abbr]
	        sents_wo_abbrv = sents_wo_abbrv.str.replace(abbr, abbr_conv)
		return sents_wo_abbrv

	'''
	Method to remove punctuations from the text. 
	'''
	def punctuations_removal(self):
		sents_wo_punct = self.updated_text.copy()
	    for punct in PUNCTUATIONS:
	        sents_wo_punct = sents_wo_punct.str.replace(punct, ' ')
	    return sents_wo_punct

	'''
	Method to remove numbers from the text.
	'''
	def numbers_removal(self):
		sents_wo_numb = self.updated_text.copy()
	    for numb in list(range(10)):
	        sents_wo_numb = sents_wo_numb.str.replace(str(numb), ' ')
	    return sents_wo_numb

	'''
	Method to remove stopwords from the text.
	'''
	def stopwords_removal(self):
	    sents_wo_stopwords = self.updated_text.copy()
	    for stopword in STOPWORDS:
	        sents_wo_stopwords = sents_wo_stopwords.str.replace(' '+stopword+' ', ' ')
	    return sents_wo_stopwords

	'''
	Method to lemmatize words in the text.
	'''
	def lemmatization(self):
		return self.updated_text.map(lambda sent: ' '.join([lemmatizer.lemmatize(word) for word in sent.split()]))

	'''
	Method to remove spaces and line feed symbols from the text.
	'''
	def spaces_removal(self):
		return self.updated_text.str.replace('\n', ' ').map(str.strip)

	######################################################################################
    ######### GETTER METHODS

	"""
	Getter method for the variable containing the original text.
	"""
	@property
	def original_text(self):
		return self.original_text

	"""
	Getter method for the variable containing the updated text.
	"""
	@property
	def updated_text(self):
		return self.updated_text	
