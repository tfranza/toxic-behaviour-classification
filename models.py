
import json
import joblib
import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

from metrics import get_metrics

FEATURES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class SklearnSupervisedModel:
    '''
    Encapsulates the basic functions :meth: `fit`, :meth: `predict` and :meth: `eval` to deal with sklearn supervised models.
        '''

    def __init__(self, model: ClassifierMixin, name: str):
        '''
        Builder. Assigns the sklearn model and the name of the model to the corresponding homonimous class attributes.
        
        :param model: Sklearn model to fit.
        :param name: Name of the model. Useful for saving and loading purposes.
        '''
        super(SklearnSupervisedModel, self).__init__()
        self.model = model
        self.name = name

    def train(self, train_X: np.array, train_Y: np.array):
        '''
        Fits the classifier to the training set. Updates the :attr: `model`.
    
        :param train_X: features of the training set.
        :param train_Y: labels associated to the features.
        '''
        print('   - begin training')
        self.trainset = (train_X, train_Y)
        self.model.fit(train_X, train_Y)
        print('   - end training')

    def eval(self, test_X: np.array, test_Y: np.array, save_eval: bool=True, verbose: bool=True) -> dict:
        '''
        Predicts labels from the features using the :attr: `model` and evaluates them against the real labels.
        
        :param test_X: features of the validation set.
        :param test_Y: labels of the validation set.
        :param save_eval: True if we want to dump the metrics.
        :attr name: name of the used sklearn model.  

        :return: metrics adopted for the evaluation task.
        '''
        self.testset = (test_X, test_Y)
        self.pred = self.model.predict(test_X)
        self.metrics = get_metrics(FEATURES, test_Y, self.pred)
        if verbose:
            for i, label in enumerate(self.metrics['name']):
                acc = self.metrics['acc'][i]
                prec = self.metrics['prec'][i]
                rec = self.metrics['rec'][i]
                f1 = self.metrics['f1'][i]
                print()
                print(f'Label: {label}')
                #print(f'Prec: {np.round(prec,5)}, Rec: {np.round(rec,5)}, F1-score = {np.round(f1,5)}')
                print('Confusion Matrix:\n', self.metrics['cm'][i])
            print('\n', self.metrics['report'])

        if save_eval:
            with open(f'results/{self.name}', 'w') as handle:
                json.dump(self.metrics, handle)
        return self.metrics

    def save_model(self):
        '''
        Saves the model.

        :attr name: name of the used sklearn model.  
        '''
        joblib.dump(self.model, f'models/{self.name}.model')

    def load_model(self, model_name: str) -> OneVsRestClassifier:
        '''
        Loads the model.
        
        :param model_name:
        
        :return: the fitted model
        '''
        self.name = model_name
        return joblib.load(f'models/{self.name}.model') 


class ModelLoader:
    
    def __init__(self, random_state):
        '''
        Builder. Assigns the random state to the corresponding attribute.
        
        :param random_state: random state to make experiments replicable.
        '''
        super(ModelLoader, self).__init__()
        
        self.random_state = random_state

    def load_model(self, model_name) -> SklearnSupervisedModel:
        '''
        Builds and returns the model corresponding to the model name passed as input.
            
        :param model_name: name of the model to build.

        :return: sklearn supervised model initialized.
        '''
        print(f'\n > LOADING MODEL: {model_name} ...')
        return eval(f'self.{model_name}()')


    ###################################################
    ##### MODELS

    def logreg(self) -> SklearnSupervisedModel:
        '''
        Builds the Logistic Regression classifier with ad-hoc chosen parameters for the available data. 
        In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
            
        :param model_name: name of the model to build.
        :attr random_state: random state to make experiments replicable.

        :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
        ''' 
        model = OneVsRestClassifier(
            LogisticRegression(
                penalty = 'l2',
                C = 9,
                solver = 'liblinear',
                class_weight = 'balanced',
                random_state = self.random_state
            )
        )
        return SklearnSupervisedModel(model=model, name='logreg')

    def svc_linear(self) -> SklearnSupervisedModel:
        '''
        Builds the Linear Support Vector classifier with ad-hoc chosen parameters for the available data. 
        In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
                
        :param random_state: random state to make experiments replicable.
        
        :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
        '''
        model = OneVsRestClassifier(
            LinearSVC(
                dual = False, 
                class_weight = 'balanced',
                random_state = self.random_state
            )
        )
        return SklearnSupervisedModel(model=model, name='svc_linear')

    def svc_poly(self) -> SklearnSupervisedModel:
        '''
        Builds the Support Vector classifier with polynomial features and ad-hoc chosen parameters for the available data. 
        In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
                
        :param random_state: random state to make experiments replicable.
        
        :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
        '''
        model = OneVsRestClassifier(
            SVC(
                kernel='poly', 
                random_state=self.random_state
            )
        )
        return SklearnSupervisedModel(model=model, name='svc_poly')

    def randomforest(self) -> SklearnSupervisedModel:
        '''
        Builds the RandomForest classifier with ad-hoc chosen parameters for the available data. 
        In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
                
        :param random_state: random state to make experiments replicable.

        :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
        '''
        model = OneVsRestClassifier(
            RandomForestClassifier(
                class_weight='balanced',
                random_state=self.random_state
            )
        )
        return SklearnSupervisedModel(model=model, name='randomforest')

    def xgboost(self) -> SklearnSupervisedModel:
        '''
        Builds the Gradient Boosting classifier with ad-hoc chosen parameters for the available data. 
        In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
                
        :param random_state: random state to make experiments replicable.

        :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
        '''
        model = OneVsRestClassifier(
            GradientBoostingClassifier(
                max_depth=9,
                random_state=self.random_state
            )
        )
        return SklearnSupervisedModel(model=model, name='xgboost')

