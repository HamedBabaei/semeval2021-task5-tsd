import random

class Random:    
    def __init__(self):
        '''
            Initialize Random Model - Define Random Model
        '''
        self.random_baseline = lambda text: [i for i, char in enumerate(text) if random.random()>0.5]

    def fit(self, X, Y):
        '''
            This method is only to keep interface definitions the same in cross validation
        '''
        self.__init__()
        pass

    def predict(self, X):
        '''
            Make a Random Predictions
        Arguments:
            X: a input dataset
        Returns:
            predict: a random predicted toxic spans
        '''

        predict = []
        for x in X:
            predict.append(self.random_baseline(x))
        return predict