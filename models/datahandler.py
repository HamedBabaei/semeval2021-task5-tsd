import pandas as pd
from ast import literal_eval

def load_train(path, verbose=False):
    '''
        Loading Train Data
    Arguments:
        path(string): path to train data
        verbose(bool): display a head or not
    Returns:
        train: loaded train data
    '''
    train = pd.read_csv(path)
    train["spans"] = train.spans.apply(literal_eval)
    if verbose:
        display(train.head(3))
    return train

