from sklearn.linear_model import LogisticRegression
import pandas as pd
import spacy
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from xgboost import XGBClassifier
from transformers import BertForSequenceClassification, BertConfig
from models.charbert.character_bert import CharacterBertModel
from transformers import BertTokenizer
from models.charbert.utils.character_cnn import CharacterIndexer

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
bert_word_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = BertForSequenceClassification(config=config)

#print(">>> wordpiece embeddings whith shape of:", bertmodel.bert.embeddings.word_embeddings)

character_bert_model = CharacterBertModel.from_pretrained('../pretrained-models/general_character_bert/')
bertmodel.bert = character_bert_model
char_indexer = CharacterIndexer()  # This converts each token into a list of character indices

#print(">>> wordpieces are replaced with a CharacterCNN with followint architecture\n:",
#      bertmodel.bert.embeddings.word_embeddings)   


__vectors__ = {}
nlp = spacy.load('en_core_web_lg')


def get_char_bert_features(text, word_by_word=False):
    #F_len = lambda X:np.array([len(x) for x in X])
    if text.isspace() or repr(text) == "'️'":
        return np.zeros((1,768))[0]
    tokenized_text = bert_word_tokenizer.basic_tokenizer.tokenize(text) #this is NOT wordpiece tokenization
    input_tensor = char_indexer.as_padded_tensor([tokenized_text])  # we build a batch of only one sequence
    #input_tensor.shape >>> torch.Size([1, 8, 50]) # (batch_size, sequence_length, character_embedding_dim)
    output = bertmodel.bert(input_tensor)[0][0]

    if word_by_word:
        return output
    else:
        vec = output.detach().numpy()#/F_len(tokenized_text).reshape(-1,1)
        return  sum(vec)/vec.shape[0]

def get_charbert_embedding_vector(word):
    try:
        vector = __vectors__[word]
    except:
        __vectors__[word] = get_char_bert_features(word)
        vector = __vectors__[word]
    return vector

def get_spacy_embedding_vector(word):
    try:
        vector = __vectors__[word]
    except:
        __vectors__[word] = nlp(word).vector
        vector = __vectors__[word]
    return vector

class MLModels:
    
    def __init__(self, model_name='xgb', use_charbert=False):
        '''
        Initialize MLModels model
        '''
        self.model_name = model_name
        if self.model_name == 'xgb':
            self.model = XGBClassifier()
        elif self.model_name == 'lr':
            self.model = LogisticRegression()
        self.func_x = lambda x:[x[i][0] for i in range(len(x))]
        self.func_y = lambda x:[x[i][2] for i in range(len(x))]
        self.use_charbert = use_charbert
        if self.use_charbert:
            print("CharBert Representation Activated")
            self.F_features = lambda word: get_charbert_embedding_vector(word)
        else:
            print("Spacy Vector Representation Activated")
            self.F_features = lambda word: list(get_spacy_embedding_vector(word))
        self.max_abs_scaler = MaxAbsScaler()
        
    def get_features(self, X):
        features = []
        for i in range(len(X)):
            for x in self.func_x(X[i]):
                features.append(self.F_features(x))
        return features
    
    def fit(self, X, Y=None):
        '''
            Train ML model
        Arguments:
            X: a train set with input format of CRF
            Y: None, just to follow the fit architecture
        Returns: None
        '''
        self.__init__(model_name=self.model_name, use_charbert = self.use_charbert)
        X_train = self.get_features(X)
        Y_train = [x for i in range(len(X)) for x in self.func_y(X[i])]
        self.max_abs_scaler.fit(X_train)
        self.model.fit(self.max_abs_scaler.transform(X_train), Y_train)
        
    def predict(self, X):
        '''
            Make Prediction using pretrained ML model
        Arguments:
            X: a test set with input format of CRF
        Returns: 
            predicts
        '''
        preds = []
        for X_ in X:
            X_test_data = [__vectors__[x] if x in __vectors__ else self.F_features(x) for x in self.func_x(X_)]
            preds.append(self.model.predict(self.max_abs_scaler.transform(X_test_data)))
        return preds
    
    def predict_proba(self, X):
        '''
            Make prediction probability using pretrained ML model
        Arguments:
            X: a test set with input format of CRF
        Returns: 
            predict_probas
        '''
        preds = []
        for X_ in X:
            X_test_data = [__vectors__[x] if x in __vectors__ else list(self.F_simple(x)) for x in self.func_x(X_)]
            preds.append(self.model.predict_proba(self.max_abs_scaler.transform(X_test_data)))
        return preds