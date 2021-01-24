import os
import nltk
from nltk import MaxentClassifier
import models.utils as utils
from nltk.stem.porter import *
import numpy as np
import spacy

from transformers import BertForSequenceClassification, BertConfig
from models.charbert.character_bert import CharacterBertModel
from transformers import BertTokenizer
from models.charbert.utils.character_cnn import CharacterIndexer
import numpy as np

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
bert_word_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bertmodel = BertForSequenceClassification(config=config)

#print(">>> wordpiece embeddings whith shape of:", bertmodel.bert.embeddings.word_embeddings)

character_bert_model = CharacterBertModel.from_pretrained('../pretrained-models/general_character_bert/')
bertmodel.bert = character_bert_model
char_indexer = CharacterIndexer()  # This converts each token into a list of character indices

#print(">>> wordpieces are replaced with a CharacterCNN with followint architecture\n:",
#      bertmodel.bert.embeddings.word_embeddings)   


path = "/usr/local/bin/megam-64"
os.environ["MEGAM"] = path
nltk.config_megam(path)

__vectors__ = {}
nlp = spacy.load('en_core_web_lg')

def get_chear_bert_features(text, word_by_word=False):
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
        __vectors__[word] = get_chear_bert_features(word)
        vector = __vectors__[word]
    return vector

def get_spacy_embedding_vector(word):
    try:
        vector = __vectors__[word]
    except:
        __vectors__[word] = nlp(word).vector
        vector = __vectors__[word]
    return vector

def get_conll_features(index, sentence, pos, chunk, use_embedding=False, use_charbert=False):
    #print(index, sentence)
    """Function used to extract features for the CoNLL dataset
    
    'w' represents word feature
    't' represents POS tag feature
    'c' represents chunk tag feature
    '-n' represents previous 'n' feature
    '+n' represents posterior 'n' feature
    """
    
    features = {}
    last_index = len(sentence) - 1
    word = sentence[index]
    word_lc = word.lower()
    
    # features from current word:
    features['w'] = word
    features['t'] = pos[index]
    features['length'] = len(word)
    features['uppercase'] = any(x.isupper() for x in word)
    features['firstletter'] = word[0].isupper() and (len(word) > 1)
    features['hasdigits'] = any(x.isdigit() for x in word)
    features['c'] = chunk[index]
    features['loc_flag'] = ('field' in word_lc) or ('land' in word_lc) or ('burgh' in word_lc) or ('shire' in word_lc) 
    features['hasdot'] = ('.' in word and len(word) > 1)
    features['endsinns'] = (len(word) > 1 and word_lc[-2:] == 'ns')
    
    if use_embedding:
        if use_charbert:
            features['embedding'] = tuple(get_charbert_embedding_vector(word))
        else:
            features['embedding'] = tuple(get_spacy_embedding_vector(word))
    
    # features from previous 2 words
    if index == 0: # first word in sentence
        features['t-2 t-1'] = '<B> <B>'
        features['t-1'] = '<B>'
        features['w-2'] = '<B>'
        features['w-1'] = '<B>'
        features['c-2 c-1'] = '<B> <B>'
        features['c-1'] = '<B>'
    elif index == 1: # second word in sentence
        features['t-2 t-1'] = '<B> ' + pos[0]
        features['t-1'] = pos[0]
        features['w-2'] = '<B>'
        features['w-1'] = sentence[0]
        features['c-2 c-1'] = '<B> ' + chunk[0]
        features['c-1'] = chunk[0]
    else:
        features['t-2 t-1'] = pos[index-2] + ' ' + pos[index-1]
        features['t-1'] = pos[index-1]
        features['w-2'] = sentence[index-2]
        features['w-1'] = sentence[index-1]
        features['c-2 c-1'] = chunk[index-2] + ' ' + chunk[index-1]
        features['c-1'] = chunk[index-1]

      
    # features from posterior 2 words
    if index == last_index: # last word in sentence
        features['t+1 t+2'] = '<E> <E>'
        features['t+1'] = '<E>'
        features['w+2'] = '<E>'
        features['w+1'] = '<E>'
    elif index == last_index - 1: # second to last word in sentence
        features['t+1 t+2'] = pos[last_index] + ' <E>'
        features['t+1'] = pos[last_index]
        features['w+2'] = '<E>'
        features['w+1'] = sentence[last_index]
    else:
        features['t+1 t+2'] = pos[index+1] + ' ' + pos[index+2]
        features['t+1'] = pos[index+1]
        features['w+2'] = sentence[index+2]
        features['w+1'] = sentence[index+1]
    
    return features

class CMM:

    def __init__(self, use_embedding=False, use_charbert=False):
        self.use_embedding = use_embedding
        self.use_charbert= use_charbert

    def fit(self, X, Y=None):
        self.__init__(use_embedding = self.use_embedding, use_charbert= self.use_charbert)
        X_train = X
        train_data = [(get_conll_features(i, [word[0] for word in X], [pos[1] for pos in X], 
                                  [chunk[2] for chunk in X], use_embedding = self.use_embedding, 
                                  use_charbert = self.use_charbert), vals[3]) 
                      for X in X_train for i, vals in enumerate(X)]
        
        self.model = MaxentClassifier.train(train_data, algorithm='MEGAM')

    def predict(self, X):
        X_test = X
        preds =[]
        for X in X_test:
            pred = []
            for i, vals in enumerate(X):
                features = get_conll_features(i,[word[0] for word in X], [pos[1] for pos in X], 
                                            [chunk[2] for chunk in X], use_embedding = self.use_embedding,
                                            use_charbert = self.use_charbert)
                pred.append(self.model.classify(features)) 
            preds.append(pred)
        return preds
