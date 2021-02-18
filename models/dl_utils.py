from tqdm import tqdm
import numpy as np
import pickle 
from transformers import GPT2Tokenizer, GPT2Model
from transformers import RobertaTokenizer, RobertaModel
from nltk.stem.porter import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

stemmer = PorterStemmer()

special_chars = ['O', '4', '͞', 'ü', '😞', '»', '”', '\x7f', '"', '7', '*', '8', ')',
      '😂', '💀', '{', '0', 'ʻ', '💨', '•', '#', '_', '😊', '😜', 'ó', '😅', 
      '¬', '☠', '🙄', '😉', '😆', '―','(', '.', '?', '😵', '💥', ':', '🆘',
      '$', '=', '+', ';', '🔥', '😁','`', 'ï', '2', 'ê', 'é', '3', '-', '🤥',
      '⚽', 'É', '️', '!', '5', '✭', '9', '😈', 'ˈ', '😬', '@', '«', '/', '▀', '’',
      '😡', '1', '%', ',', '&', '6', '\\', ']', "'", '[', '^', '}', '⚾', '\u200b',
       '☭', '☹', '<', 'l', '™', '👎']

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

GPT2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
GPT2_model = GPT2Model.from_pretrained('gpt2')

def load_pkl(path):
    '''load pickle data from specified path'''
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    return pkl

def preprocessing(X, clean=False):
    cleaned_x = []
    for words in tqdm(X):
        cleaned_words = []
        for word in words :
            w, pos, tag, span = word
            if clean:
                w = ''.join([_w for _w in w if _w not in special_chars])
            if len(w) != 0:
                cleaned_words.append((w.lower(), pos, tag, span))
        cleaned_x.append(cleaned_words)
    return cleaned_x

def load_glove(path):
    """Loading glove vectors (glove.840B.300d.txt)"""
    embeddings_index = {}
    f = open(path)
    for line in tqdm(f):
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()
    return embeddings_index

def glove_preprocessor(X):
    return stemmer.stem(''.join([w for w in X if w not in special_chars ]).lower())

def get_glove_vector(glove, word):
    glove_vector = glove.get(glove_preprocessor(word))
    if glove_vector is None:
        glove_vector = np.zeros(300)
    return glove_vector

def get_glove_features(word_index, glove, embed_size=300):
    embeddings = np.zeros((len(word_index), embed_size))
    oov = 0
    for word, i in tqdm(word_index.items()):
        glove_vector = glove.get(glove_preprocessor(word))
        if glove_vector is None:
            glove_vector = np.zeros(300)
            oov += 1
        embeddings[i] = glove_vector
    print("GLOVE:::OOV:{}/{} ".format(oov, len(word_index)))
    return embeddings

def get_gpt2_vector(gpt2, word):
    try:
        return gpt2[word]
    except:
        inputs = GPT2_tokenizer(word, return_tensors="pt")
        outputs = GPT2_model(**inputs)
        vec = np.average(outputs[0].detach().numpy(), axis=1)[0]
        gpt2[word] = vec
        return vec

def get_gpt2_features(word_index, gpt2, embed_size=768):
    embeddings = np.zeros((len(word_index), embed_size))
    for word, i in tqdm(word_index.items()):
        gpt2_vector = get_gpt2_vector(word)
        embeddings[i] = gpt2_vector
    return embeddings

def get_roberta_vector(roberta, word):
    try:
        return roberta[word]
    except:
        inputs = roberta_tokenizer(word, return_tensors="pt")
        outputs = roberta_model(**inputs)
        vec = np.mean(outputs[0].detach().numpy(),  axis=1)[0]
        roberta[word] = vec
        return vec

def get_roberta_features(word_index, roberta, embed_size=768):
    embeddings = np.zeros((len(word_index), embed_size))
    for word, i in tqdm(word_index.items()):
        roberta_vector = get_roberta_vector(word)
        embeddings[i] = roberta_vector
    return embeddings

def ensemble_features(word_index, glove, gpt2, roberta, embed_size=1068):
    embeddings = np.zeros((len(word_index), embed_size))
    for word, i in tqdm(word_index.items()):
        gpt2_vector = get_gpt2_vector(gpt2, word)
        roberta_vector = get_roberta_vector(roberta, word)
        glove_vector = get_glove_vector(glove, word)
        embeddings[i] = np.concatenate([gpt2_vector+roberta_vector, glove_vector])
    return embeddings

def get_indecies(words_list):
    #words_list = X_train + X_test_clean+X_test_fn
    words = set([w for x in words_list for w, tag, label, span in x])
    tags = set([tag for x in word_list for w, pos, tag, span in x])
    n_tags = len(tags)
    n_words = len(words)
    print("n_tags:{}, n_words:{}\n".format(n_tags, n_words))

    max_len = max([len(x) for x in words_list])
    print("max_len:{} ".format(max_len))

    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    
    idx2word = {i: w for w, i in word2idx.items()}
    
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}

    print("word2idx['fuck']:{},  tag2idx['normal']:{}\n".format(word2idx["fuck"], 
                                                                tag2idx["normal"]))
    max_features = len(word2idx)
    print("Maximum features:{}".format(max_features))
    return word2idx, idx2word, tag2idx, idx2tag, max_len, max_features

def padsequences(X, word2idx, tag2idx, max_len):
    X_word2idx = [[word2idx[w[0]] if w[0] in word2idx else word2idx["UNK"] for w in s] 
                  for s in tqdm(X)]
    X_seq = pad_sequences(maxlen=max_len, sequences=X_word2idx, 
                          value=word2idx["PAD"], padding='post', truncating='post')

    y_tag2idx = [[tag2idx[w[2]] for w in s] for s in X]
    y_seq = pad_sequences(maxlen=max_len, sequences=y_tag2idx, 
                          value=tag2idx["PAD"], padding='post', truncating='post')
    return X_seq, y_seq

def get_lstm_layers(number=1):
    pass

def get_lstm_attention_layers(number=1):
    pass

def get_gru_layers(number=1):
    pass

def get_gru_attention_layers(number=1):
    pass

def get_prediction_spans(predicts, X_seq, idx2tag):
    predictions, probas = [], []
    for i in tqdm(range(predicts.shape[0])):
        proba = np.argmax(predicts[i], axis=-1)
        preds, lenght = [],0
        for w, pred in zip(X_seq[i], proba):
            if w != 0:
                preds.append(idx2tag[pred])
                lenght+=1
        predictions.append(preds)
        probas.append(predicts[i][:lenght])
    return predictions, probas

