
import numpy as np
from skmultilearn.problem_transform import LabelPowerset

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import CondensedNearestNeighbour, ClusterCentroids, NearMiss, RandomUnderSampler

class Augmentator:

    def __init__(self, X:np.array, y:np.array):
        '''
        Builder. Assigns the features and the labels to the homonymous class attributes. Initializes the augmentation method to None.
        
        :param X: features of the dataset to be augmented.
        :param y: labels of the dataset to be augmented.
        '''
        super(Augmentator, self).__init__()

        self.X = X
        self.y = y
        self.method = None

    def augment(self, approach:str, resampling_type:str, resampling_method:str) -> tuple:
        '''
        Resamples the stored dataset, affected by labels imbalance.
        
        :param approach: kind of approach to tackle the multilabel imbalance problem. Current options: label_powerset.
        :param resampling_type: undersampling or oversampling flag.
        :param resampling_method: name of the resampling method falling under the :attr resampling_type:.  

        :return: the resampled dataset as the couple of :attr X: and :attr y. 
        '''
        print(f'   - applying {resampling_type} with {resampling_method} ...')
        
        try:
            eval(f'self.{resampling_type}(\'{resampling_method}\')')
        except:
            print(f' - either {resampling_type} or {resampling_method} not found as an option available for selection.')

        if self.method:
            if approach=='label_powerset':
                lp = LabelPowerset()
                X, y = self.method.fit_resample(self.X, lp.transform(self.y))
                return X.A, lp.inverse_transform(y).A
            else:
                return None, None
        else:
            return None, None


    ###################################################
    ##### AUGMENTATION OPERATIONS

    def undersampling(self, method_name:str):
        '''
        Applies the undersampling operation over the stored dataset using the method name passed as input.
        
        :param method_name: name of the undersampling method to apply on the dataset. 
        '''
        if method_name == 'random':
            self.method = RandomUnderSampler(sampling_strategy='majority')
        elif method_name == 'nearmiss':
            self.method = NearMiss(sampling_strategy='majority', version=2)
        elif method_name == 'cluster_centroids':
            self.method = ClusterCentroids(sampling_strategy='majority')
        elif method_name == 'condensed_nn':
            self.method = CondensedNearestNeighbour(sampling_strategy='majority')
        else:
            self.method = None

    def oversampling(self, method_name:str):
        '''
        Applies the oversampling operation over the stored dataset using the method name passed as input.
        
        :param method_name: name of the oversampling method to apply on the dataset. 
        '''
        if method_name == 'random':
            self.method = RandomOverSampler(sampling_strategy='not majority')
        else:
            self.method = None

