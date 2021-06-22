
import numpy as np
import pickle
import requests

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

BADWORDS_LINK = 'https://www.cs.cmu.edu/~biglou/resources/bad-words.txt'    # common toxic badwords
MAX_COUNT_FEATURES = 10000

class FeaturesExtractor:
    '''
    Class for dealing with the extraction of engineered features from the input text.
    '''
    
    def __init__(self, text: np.array, test: bool = False):
        '''
        Builder. Loads the numpy representation of the text and initializes :attr: `features` for the engineered features.
        
        :param text: textual source which to extract the features from.
        :param test: flag to indicate whether we're working on the test set or not.
        '''
        super(FeaturesExtractor, self).__init__()

        self.text = text
        self.test = test

    def extract(self, features: list):
        """
        Applies the feature engineering operations to the loaded data. These operations are given in input and given 
        their order of appearance in the list.

        :param features: list of features to extract from the textual source.
        """
        if not features:
            print('\n > FEATURE ENGINEERING > aborted: empty features list!')
            return self
        else:
            print('\n > FEATURE ENGINEERING')

        self.features = np.zeros(shape=(self.text.shape[0],1))
        for feature in features:
            print(f'   - {feature} extraction ...')
            if feature == 'n_badwords':
                self.extract_n_badwords()
            elif feature == 'sent_len':
                self.extract_sentence_length()
            elif feature == 'n_!':
                self.extract_n_exclamation_marks()
            elif feature == 'n_?':
                self.extract_n_interrogation_marks()
            elif feature == 'n_upper_words':
                self.extract_n_upper_words()
            elif feature == 'n_upper_letters':
                self.extract_n_upper_letters()
            elif feature == 'word_counts':
                self.extract_word_counts_tfidf('counts')
            elif feature == 'word_tfidf':
                self.extract_word_counts_tfidf('tfidf')
            else:
                print(f' - {feature} not found as an option available for selection.')


    ###################################################
    ##### FEATURE ENGINEERING OPERATIONS

    def extract_n_badwords(self):
        '''
        Extracts word counts that involve explicit offense or that have toxicity in them. Loads the html list and sums the amount of 
        occurrences of badwords as a whole for each sentence.

        :attr text: textual source which to extract the feature from.
        :attr features: engineered feature matrix to be updated with the new features.
        '''
        html_doc = requests.get(BADWORDS_LINK)
        badwords = html_doc.text.split('\n')[1:-1]
        f = lambda x: sum(1 for word in x.split(' ') if word in badwords)
        new_features = np.array([f(x) for x in self.text])
        self.features = np.append(self.features, new_features) 

    def extract_sentence_length(self):
        '''
        Extracts the sentence length as a measure of the number of letters.

        :attr text: textual source which to extract the feature from.
        :attr features: engineered feature matrix to be updated with the new features.
        '''
        f = lambda x: len(x)
        new_features = np.array([f(x) for x in self.text])
        self.features = np.append(self.features, new_features) 

    def extract_n_exclamation_marks(self):
        '''
        Extracts the amount of exclamation marks in the sentence.

        :attr text: textual source which to extract the feature from.
        :attr features: engineered feature matrix to be updated with the new features.
        '''
        f = lambda x: len(x.split('!'))-1
        new_features = np.array([f(x) for x in self.text])
        self.features = np.append(self.features, new_features) 

    def extract_n_interrogation_marks(self):
        '''
        Extracts the amount of interrogation marks in the sentence.

        :attr text: textual source which to extract the feature from.
        :attr features: engineered feature matrix to be updated with the new features.
        '''
        f = lambda x: len(x.split('?'))-1
        new_features = np.array([f(x) for x in self.text])
        self.features = np.append(self.features, new_features) 

    def extract_n_upper_words(self):
        '''
        Extracts the amount of upper words in the sentence.

        :attr text: textual source which to extract the feature from.
        :attr features: engineered feature matrix to be updated with the new features.
        '''
        f = lambda x: sum(1 for word in x.split(' ') if word.isupper())
        new_features = np.array([f(x) for x in self.text])
        self.features = np.append(self.features, new_features) 

    def extract_n_upper_letters(self):
        '''
        Extracts the amount of upper letters in the sentence.

        :attr text: textual source which to extract the feature from.
        :attr features: engineered feature matrix to be updated with the new features.
        '''
        f = lambda x: sum(1 for letter in x if letter.isupper())
        new_features = np.array([f(x) for x in self.text])
        self.features = np.append(self.features, new_features) 

    def extract_word_counts_tfidf(self, feature_type: str = 'counts'):
        '''
        Method that extracts the word counts/tfidf for each word in the data. The amount of words is limited by the 
        MAX_COUNT_FEATURES variable. The vectorizer will try to fit only on the training data.
        
        :param feature_type: type of feature to extract from the text. Allowed values: 'counts', 'tfidf'

        :attr text: textual source which to extract the feature from.
        :attr features: engineered feature matrix to be updated with the new features.
        '''
        vectorizer = None 
        word_features = None
        try:		# vectorizer already generated		
            with open(f'resources/{feature_type}_vectorizer.pickle', 'rb') as handle:
                vectorizer = pickle.load(handle)
            word_features = vectorizer.transform(self.text)
        except:		# vectorizer not yet generated
            if self.test:
                log.info(f' - {feature_type} vectorizer not found!')		
            else:
                if feature_type=='counts':
                    vectorizer = CountVectorizer(analyzer='word', max_features=MAX_COUNT_FEATURES)
                elif feature_type=='tfidf':
                    vectorizer = TfidfVectorizer(analyzer='word', max_features = MAX_COUNT_FEATURES)
                word_features = vectorizer.fit_transform(self.text)
                with open(f'resources/{feature_type}_vectorizer', 'wb') as handle:
                    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        new_features = word_features
        self.features = np.append(self.features, new_features) 
