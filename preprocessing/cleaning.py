
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

STOPWORDS = set(stopwords.words('english'))

ABBREVIATIONS_MAPPING = {
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
    '''
    Class for dealing with the cleaning of the data: makes available several methods to do it. 
    '''

    def __init__(self, text: str):
        '''
        Builder. Assigns the textual source to both original and updated_text (in the latter one it will be just a beginning point).
        
        :param text: text source to be cleaned.
        '''
        super(TextCleaner, self).__init__()

        self.original_text = text
        self.updated_text = text						

    def brush(self, operations: list):
        """
        Applies the cleaning operations to the loaded data. The cleaning operations are given in input and given their order of appearance 
        in the list.

        :param operations_list: list of cleaning operations to apply on text.
        """
        self.operations = operations

        if not operations:
            print('\n > TEXT CLEANING > aborted: empty operations list!')
            return
        else:
            print('\n > TEXT CLEANING')

        for operation in self.operations:
            print(f'   - applying {operation} ...')
            if operation == 'lowering':
                self.lowering()
            elif operation == 'abbreviations_removal':
                self.abbreviations_removal()
            elif operation == 'punctuations_removal':
                self.punctuations_removal()
            elif operation == 'numbers_removal':
                self.numbers_removal()
            elif operation == 'stopwords_removal':
                self.stopwords_removal()
            elif operation == 'lemmatization':
                self.lemmatization()
            elif operation == 'spaces_removal':
                self.spaces_removal()
            else:
                print(f' - {operation} not found as an option available for selection.')

    ###################################################
    ##### CLEANING OPERATIONS

    def lowering(self):
        '''
        Transforms every uppercased letter into a lowercased one. 

        :attr updated_text: text cleaned in consecutive steps. Initialized as the raw text.
        '''
        self.updated_text = self.updated_text.map(str.lower)

    def abbreviations_removal(self, mapping: dict=ABBREVIATIONS_MAPPING):
        '''
        Removes abbreviations from the :attr: `updated_text` using the ABBREVIATIONS mapping. 

        :param mapping: dictionary containing the mapping of each abbreviation with the corresponding substitutive phrase.

        :attr updated_text: text cleaned in consecutive steps. Initialized as the raw text.
        '''
        sents_wo_abbrv = self.updated_text.copy()
        for abbr in mapping:
            abbr_conv = mapping[abbr]
            sents_wo_abbrv = sents_wo_abbrv.str.replace(abbr, abbr_conv)
        self.updated_text = sents_wo_abbrv

    def punctuations_removal(self, punctuations: list=PUNCTUATIONS):
        '''
        Removes the selected punctuations from the :attr: `updated_text`. 

        :param punctuations: list of punctuations to remove.

        :attr updated_text: text cleaned in consecutive steps. Initialized as the raw text.
        '''
        sents_wo_punct = self.updated_text.copy()
        for punct in punctuations:
            sents_wo_punct = sents_wo_punct.str.replace(punct, ' ')
        self.updated_text = sents_wo_punct

    def numbers_removal(self):
        '''
        Removes the numbers from the :attr: `updated_text`.

        :attr updated_text: text cleaned in consecutive steps. Initialized as the raw text.
        '''
        sents_wo_numb = self.updated_text.copy()
        for numb in list(range(10)):
            sents_wo_numb = sents_wo_numb.str.replace(str(numb), ' ')
        self.updated_text = sents_wo_numb

    def stopwords_removal(self):
        '''
        Removes the stopwords from the :attr: `updated_text`.

        :attr updated_text: text cleaned in consecutive steps. Initialized as the raw text.
        '''
        sents_wo_stopwords = self.updated_text.copy()
        for stopword in STOPWORDS:
            sents_wo_stopwords = sents_wo_stopwords.str.replace(' '+stopword+' ', ' ')
        self.updated_text = sents_wo_stopwords

    def lemmatization(self):
        '''
        Lemmatizes words in the :attr: `updated_text`.

        :attr updated_text: text cleaned in consecutive steps. Initialized as the raw text.
        '''
        self.updated_text = self.updated_text.map(lambda sent: ' '.join([lemmatizer.lemmatize(word) for word in sent.split()]))

    def spaces_removal(self):
        '''
        Removes spaces and line feed symbols from the :attr: `updated_text`.

        :attr updated_text: text cleaned in consecutive steps. Initialized as the raw text.
        '''
        self.updated_text = self.updated_text.str.replace('\n', ' ').map(str.strip)



