
from sklearn.model_selection import train_test_split

from preprocessing.preprocessing import Preprocessor
from models import ModelLoader

def train():

    random_state = 0
    model_name = 'logreg'

    prep_cleaning_steps = [
        'lowering',
        #'abbreviations_removal',
        #'punctuations_removal',
        #'numbers_removal',
        #'stopwords_removal',
        #'lemmatization',
        'spaces_removal'
    ]
    prep_features_to_be_engineered = [
        'word_tfidf',
        'n_badwords'
    ]
    
    # loading training set
    preprocessor = Preprocessor(filename='train.csv', pathfile='data/')
    X, y = preprocessor.preprocess(prep_cleaning_steps, prep_features_to_be_engineered)
    #test_X, test_Y = Preprocessor(filename='test.csv', pathfile='data/').preprocess(prep_cleaning_steps, prep_features_to_be_engineered)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=random_state)
    
    model = ModelLoader(random_state).load_model(model_name)
    
    model.train(X, y)
    model.eval(test_X, test_y, save_eval=True)
    model.save_model()

train()