
from preprocessing.preprocessing import Preprocessor

def train():
	
    prep_cleaning_steps = [
        'lowering'#,
#        'abbreviations_removal',
#        'punctuations_removal',
#        'numbers_removal',
#        'stopwords_removal',
#        'lemmatization',
#        'spaces_removal'
    ]
    prep_features_to_be_engineered = [
        'word_tfidf'
    ]
    
    # loading training set
    train_X, train_Y = Preprocessor(filename='train.csv', pathfile='data/').preprocess(prep_cleaning_steps, prep_features_to_be_engineered)
    test_X, test_Y = Preprocessor(filename='test.csv', pathfile='data/').preprocess(prep_cleaning_steps, prep_features_to_be_engineered)

    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)