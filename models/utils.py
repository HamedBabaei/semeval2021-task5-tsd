import json
import pickle
import pandas as pd
import os

def save_json(path, data):
    '''save json data into specified path'''
    json.dump(data, open(path, "w"), indent=4)

def load_json(path):
    '''load json data from specified path'''
    return json.load(open(path, "r"))

def save_pkl(path, data):
    '''save pickle data into specified path '''
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    '''load pickle data from specified path'''
    with open(path, "rb") as f:
        pkl = pickle.load(f)
    return pkl

def save_df(path, data):
    '''save pandas dataframe into specified path, csv'''
    data.to_csv(path, index=False)

def load_df(path):
    '''save pandas dataframe into specified path, csv'''
    return pd.read_csv(path)

def save_text(path, data):
    '''save text file into specified path'''
    open(path, "w").write(data)

def load_text(path):
    '''read text from specified path'''
    with open(path, "r") as f:
        text = f.read()
    return text

def mkdir(path):
    """create directory of it is not exist"""
    if not os.path.exists(path):
        os.mkdir(path)
