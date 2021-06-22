
import numpy as np
import pandas as pd
import requests

from preprocessing.engineering import FeaturesExtractor
from preprocessing.cleaning import TextCleaner

class Preprocessor:
    '''
    Class for dealing with the general preprocessing of the data. It stores preprocessing methods 
    that can recall either the cleaning or the feature engineering process. 
    '''

    def __init__(self, filename: str, pathfile: str = 'data/'):
        '''
        Builder. Assigns the sklearn model and the name of the model to the corresponding homonimous class attributes.
        
        :param filename: filename of the data to be processed.
        :param pathfile: path where the filename can be found.
        '''
        super(Preprocessor, self).__init__()

        print(f'\n > LOADING DATA from {pathfile}{filename} ...')

        self.df = pd.read_csv(pathfile + filename, index_col=0)
        self.text = self.df.iloc[:,0]
        labels = self.df.iloc[:,1:]						
        self.labels = labels.values if labels.shape[1]!=0 else None


    ###################################################
    ##### PREPROCESSING OPERATIONS

    def preprocess(self, cleaning_operations:list, features_to_extract:list) -> tuple:
        '''
        Preprocesses the text going through both text cleaning and feature engineering.

        :param cleaning_operations: list of cleaning operations to apply on the data.
        :param features_to_extract: list of features to extract from the textual source.

        :attr features: the engineered features after having cleaned the dataset.
        :attr labels: the labels.

        :return: the preprocessed dataset.
        '''
        self.apply_text_cleaning(cleaning_operations)
        self.apply_feature_engineering(features_to_extract)
        return self.features, self.labels

    def apply_text_cleaning(self, operations: list):
        '''
        Generates an instance of the `TextCleaner` class and applies text cleaning on the loaded data using the list of operations.

        :param operations: list of cleaning operations to apply on the data.
        :attr text: text to clean.
        '''
        cleaner = TextCleaner(self.text)
        cleaner.brush(operations)
        self.text = cleaner.updated_text

    def apply_feature_engineering(self, features: list):
        '''
        Generates an instance of the `FeatureExtractor` class and applies feature extraction on the loaded data using the list of features.

        :param features: list of features to extract from the data.
        :attr text: text which to extract features from.
        '''
        extractor = FeaturesExtractor(self.text)
        extractor.extract(features)
        self.features = extractor.features


