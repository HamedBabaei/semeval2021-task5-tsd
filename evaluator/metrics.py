
def precision(gold, predictions):
    '''
        Caclulate precision
    Arguments:
        gold(list): list of truth offensive characters for a sample
        predictions(list): list of prediction offensive characters for input sample
    Returns:
        precision
    '''
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    return len(set(gold).intersection(set(predictions)))/len(set(predictions))
        
def recall(gold, predictions):
    '''
        Caclulate recall
    Arguments:
        gold(list): list of truth offensive characters for a sample
        predictions(list): list of prediction offensive characters for input sample
    Returns:
        recall
    '''
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    return len(set(gold).intersection(set(predictions)))/len(set(gold))

def f1_score(gold, predictions):
    """
        F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
        >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    Arguments:
        predictions: a list of predicted offsets
        gold: a list of offsets serving as the ground truth
    Returns: 
        F1 Score: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)

def evaluate(gold, predictions):
    '''
        Caclulate precision recall, and f1 for all predictions
    Arguments:
        gold(2d array(in a list format)): a 2-D array of truth offensive characters for a sample
        predictions(2d array(in a list format)): a 2-D array of prediction offensive characters for input sample
    Returns:
        f1, p, r
    '''
    p = sum([precision(gold[i], predictions[i]) for i in range(len(gold))])/len(predictions)
    r = sum([recall(gold[i], predictions[i]) for i in range(len(gold))])/len(predictions)
    f1 = sum([f1_score(gold[i], predictions[i]) for i in range(len(gold))])/len(predictions)
    return f1, p, r
