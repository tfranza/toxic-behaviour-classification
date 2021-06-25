
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report 

def get_metrics(labels: list, y_true: np.array, y_pred: np.array) -> dict:
    '''
    Extracts the metrics according to the true labels and the predictions obtained from the classifier.

    :param labels: name of the labels to be evaluated
    :param y_true: matrix of labels to be matched (belonging to the validation set)
    :param y_pred: matrix of matching labels (that the classifier has predicted) 
    
    :return: dictionary of metrics. Each entry value is a list of floats with length equal to the amount of labels. 
    '''    
    name, acc, cm, rec, prec, f1 = [], [], [], [], [], []
    for i, label in enumerate(labels): 
        name = name + [label]
        acc = acc + [accuracy(y_true[:,i], y_pred[:,i])]
        cm = cm + [confusion_matrix(y_true[:,i], y_pred[:,i]).tolist()]
        rec = rec + [recall(y_true[:,i], y_pred[:,i])]
        prec = prec + [precision(y_true[:,i], y_pred[:,i])]
        f1 = f1 + [f1_score(y_true[:,i], y_pred[:,i])]
    report = classification_report(y_true, y_pred, target_names=labels)

    return dict({
        'name': name,
        'acc': acc,
        'rec': rec,
        'prec': prec,
        'f1': f1,
        'cm': cm,
        'report': report
    })
