
import json
import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC, SVC

from ./metrics import get_metrics

FEATURES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def logreg(self, random_state: int=0) -> SklearnSupervisedModel:
    '''
    Builds the Logistic Regression classifier with ad-hoc chosen parameters for the available data. 
    In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
        
    :param random_state: random state to make experiments replicable.

    :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
    '''
        
    model = OneVsRestClassifier(
        LogisticRegression(
            penalty = 'l2',
            C = 9,
            solver = 'liblinear',
            class_weight = 'balanced',
            random_state = random_state
        )
    )
    return SklearnSupervisedModel(model=model, name='logreg')

def svc_linear(self, random_state: int=0) -> SklearnSupervisedModel:
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
            random_state = random_state
        )
    )
    return SklearnSupervisedModel(model=model, name='svc_linear')

def svc_poly(self, random_state: int=0) -> SklearnSupervisedModel:
    '''
    Builds the Support Vector classifier with polynomial features and ad-hoc chosen parameters for the available data. 
    In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
            
    :param random_state: random state to make experiments replicable.
    
    :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
    '''
    model = OneVsRestClassifier(
        SVC(
            kernel='poly', 
            random_state=random_state
        )
    )
    return SklearnSupervisedModel(model=model, name='svc_poly')

def randomforest(self, random_state: int=0) -> SklearnSupervisedModel:
    '''
    Builds the RandomForest classifier with ad-hoc chosen parameters for the available data. 
    In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
            
    :param random_state: random state to make experiments replicable.

    :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
    '''
    model = OneVsRestClassifier(
        RandomForestClassifier(
            class_weight='balanced',
            random_state=random_state
        )
    )
    return SklearnSupervisedModel(model=model, name='randomforest')

def xgboost(self, random_state: int=0) -> SklearnSupervisedModel:
    '''
    Builds the Gradient Boosting classifier with ad-hoc chosen parameters for the available data. 
    In order to deal with the multilabel task, it uses OneVsRestClassifier on top of the chosen model.
            
    :param random_state: random state to make experiments replicable.

    :return: sklearn supervised model encapsulated into the SklearnSupervisedModel class.
    '''
    model = OneVsRestClassifier(
        GradientBoostingClassifier(
            max_depth=9,
            random_state=random_state
        )
    )
    return SklearnSupervisedModel(model=model, name='xgboost')

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
        super().__init__(self)
        self.model = model
        self.name = name

        def fit(self, train_X: np.array, train_Y: np.array):
        '''
        Fits the classifier to the training set. Updates the :attr: `model`.
    
        :param train_X: features of the training set.
        :param train_Y: labels associated to the features.
        '''
        self.trainset = (train_X, train_Y)
        self.model.fit(train_X, train_Y)

    def predict(self, test_X: np.array):
        '''
        Predicts labels from the features using the :attr: `model`.
        
        :param test_X: features of the validation set.
        :attr model: trained sklearn model.
        '''
        self.testset = (test_X, None)
        self.pred = self.model.predict(test_X)

    def eval(self, test_Y: np.array, save_eval: bool, verbose: bool=True) -> dict(str):
        '''
        Evaluates the predictions obtained from the classifier with the real labels.
        
        :param test_Y: labels of the validation set.
        :param save_eval: True if we want to dump the metrics.
        :attr name: name of the used sklearn model.  

        :return: metrics adopted for the evaluation task.
        '''
        self.testset = (self.testset[0], test_Y)
        self.metrics = get_metrics(FEATURES, test_Y, self.pred)

        if verbose:
            print(f' \
                Feature = {self.metrics['name']}, \
                Accuracy = {np.round(self.metrics['acc'],5)}, \
                Precision = {np.round(self.metrics['prec'],3)}, \
                Recall = {np.round(self.metrics['rec'],3)}, \
                F1-score = {np.round(self.metrics['f1'],3)}'
            )
            print('Confusion Matrix is:')
            print(self.metrics['cm'], '\n')

        if save_eval:
            with open(f'results/{self.name}', 'w') as handle:
                json.dump(self.metrics, handle)
        return self.metrics

    def save(self):
        '''
        Saves the model.

        :attr name: name of the used sklearn model.  
        '''
        with open(f'models/{self.name}.model', 'w') as handle:
            pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)


