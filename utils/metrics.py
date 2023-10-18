from sklearn.metrics import f1_score

def get_accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    accuracy = sum(y_true == y_pred) / len(y_true)
    return accuracy

def get_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def get_full_f1(y_true, y_pred, num_labels=None):
    if num_labels is None:
        num_labels = len(set(y_true))
    scores = {}
    for i in range(num_labels):
        scores[i] = f1_score(y_true, y_pred, average='binary', pos_label=i)
    scores['weighted'] = f1_score(y_true, y_pred, average='weighted')
    scores['micro'] = f1_score(y_true, y_pred, average='micro')
    scores['macro'] = f1_score(y_true, y_pred, average='macro')
    
    return scores