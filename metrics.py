
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import f1_score

def get_metrics(labels: list(str), y_true: np.array, y_pred: np.array) -> dict:
    '''
    Extracts the metrics according to the true labels and the predictions obtained from the classifier.

    :param labels: name of the labels to be evaluated
    :param y_true: matrix of labels to be matched (belonging to the validation set)
    :param y_pred: matrix of matching labels (that the classifier has predicted) 
    
    :return: dictionary of metrics. Each entry value is a list of floats with length equal to the amount of labels. 
    '''
    metrics = {
        'name': []
        'acc': [],
        'rec': [],
        'prec': [],
        'f1': []
        'cm': [],
    }
    for i, label in enumerate(labels): 
        acc   = accuracy(y_true[:,i], y_pred[:,i]) 
        cmlog = confusion_matrix(y_true[:,i], y_pred[:,i])
        rec   = recall(y_true[:,i], y_pred[:,i])
        prec  = precision(y_true[:,i], y_pred[:,i])
        f1    = f1_score(y_true[:,i], y_pred[:,i])

        metrics['name'] = metrics['name'].append(label)
        metrics['acc'] = metrics['acc'].append(acc)
        metrics['cmlog'] = metrics['cmlog'].append(cmlog)
        metrics['rec'] = metrics['rec'].append(rec)
        metrics['prec'] = metrics['prec'].append(prec)
        metrics['f1'] = metrics['f1'].append(f1)

    return metrics
