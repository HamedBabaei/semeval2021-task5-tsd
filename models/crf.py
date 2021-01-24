import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
#import spacy
#__vectors__ = {}

#nlp = spacy.load('en_core_web_lg')

#def get_feature(word):    
#    try:
#        return __vectors__[word]
#    except:
#        feature = nlp(word).vector
#        __vectors__[word] = feature
#        return feature

def word2features(sent, i):
    '''
        Making words 2 features for CRF
    Argument:
        Sent: Sentence Getter result, [[(phrase, pattern, class), .... ], ...]
        i: index of sentence
    Returns:
        features: in format of CRF input
    '''
    word = sent[i][0]
    postag = sent[i][1]

    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True


    return features

def sent2features(sent):
    '''
        Creating feature sets for sentences
    Arguments:
        sent: a sentence getter result
    Returns:
        input of CRF model for patterns of sentence i-th
    '''
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    '''
        Creating sentence labels
    Arguments:
        sent: a sentence getter result
    Returns:
        labels
    '''
    return [label for token, postag, label,span in sent]

def sent2tokens(sent):
    '''
        Creating list of tokens for sentence
    Arguments:
        sent: a sentence getter result
    Returns:
        list of tokens for sentence
    '''
    return [token for token, postag, label, span in sent]

class CRF:
    
    def __init__(self):
        '''
        Initialize CRF model
        '''
        self.model = sklearn_crfsuite.CRF(algorithm='lbfgs',
                                          max_iterations=200)

    def fit(self, X, Y=None):
        '''
            Train CRF model
        Arguments:
            X: a train set with input format of CRF
            Y: None, just to follow the fit architecture
        Returns: None
        '''
        self.__init__()
        x_train = [sent2features(s) for s in X]
        y_train = [sent2labels(s) for s in X]
        self.model.fit(x_train, y_train)
        
    def get_model(self):
        '''
            Get CRF model
        Arguments: None
        Returns:
            self.model: a pretrained crf model
        '''
        return self.model
    
    def predict(self, X):
        '''
            Make Prediction using pretrained CRF model
        Arguments:
            X: a test set with input format of CRF
        Returns: 
            predicts
        '''
        x_test = [sent2features(s) for s in X]
        return self.model.predict(x_test)
    
    def predict_proba(self, X):
        '''
            Make prediction probability using pretrained CRF model
        Arguments:
            X: a test set with input format of CRF
        Returns: 
            predict_probas
        '''
        x_test = [sent2features(s) for s in X]
        return self.model.predict_marginals(x_test)        

    def train_evaluate(self, train, test):
        '''
            fit, predict, and evaluate CRF model
        Arguments:
            train: a train set with input format of CRF
            test: a test set with input format of CRF
        Returns: None
        '''
        self.__init__()
        self.fit(train)
        y_pred = self.predict(test)
        Y_test = [sent2labels(s) for s in test]
        print(metrics.flat_classification_report(Y_test, y_pred))
        print("Sequence Accuracy:",metrics.sequence_accuracy_score(Y_test, y_pred))
        print("Accuracy:",metrics.flat_accuracy_score(Y_test,y_pred))
        print("F1-Score: ", metrics.flat_f1_score(Y_test, y_pred, average='macro'))