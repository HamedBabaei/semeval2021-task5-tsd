from tqdm import tqdm
from sklearn.model_selection import KFold
from scipy.stats import sem
import numpy as np
from tqdm import tqdm

def cross_validate(train, datamodel, model, evaluator, output_maker, 
                 cv=5, print_results=False, dataset_logger=False, 
                 calculate_average=False, return_folds=True):
    '''
        Cross Validator
    Arguments:
        train: a train dataset in df fromat
        datamodel: data handler model for model
        model: model to be cross validated
        evaluator: evaluation metrics
        output_maker: model output transition for evaluator
        cv: a cross validation fold
        print_results: to print each fold F1, P, and R or not
        dataset_logger: to display how much will take dataset prepration for train or not
        calculate_average: to calculate averaged result of F1, R, and P
    Returns:
        cv_f1: cross validation f1 scores
        cv_p: cross validation P scores
        cv_r: corss validation R scores
    '''
    logger_cv, kf = 0, KFold(n_splits=cv)
    cv_f1, cv_p, cv_r = [], [], []
    for train_idx, test_idx in tqdm(kf.split(train)):    
        logger_cv += 1
        if print_results and dataset_logger:
            print("Cross-validation fold-", logger_cv)
        elif print_results:
            print("Cross-validation fold-", logger_cv, end='  ')

        X_train, Y_train, T_train, _ = load_data(train, train_idx, datamodel, dataset_logger)
        X_test, Y_test, T_test, test_texts = load_data(train, test_idx, datamodel, dataset_logger)

        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        predictions = output_maker(X_test, test_texts, preds)
        f1, p, r = evaluator.evaluate(gold = Y_test, predictions = predictions)
        if print_results:
            print("F1:{}, \t P:{}, \t R:{}".format(f1, p, r))
        cv_f1.append(f1)
        cv_p.append(p)
        cv_r.append(r)
    if calculate_average:
        print("---------------------------------------------------------------------------")
        print(str(cv) + "-Fold Cross Validation Averaged Results:")
        average(cv_f1, cv_p, cv_r)
        print("---------------------------------------------------------------------------")
    if return_folds:
        return cv_f1, cv_p, cv_r

def load_data(train, idx, datamodel, logger):
    '''
        Loading Dataset
    Arguments:
        train: a train dataset
        idx: indexs of data to be taken
        datamodel: a model to transform data for model
        logger: to display model transformations steps and how much time it taking using tqdm
    Returns:
        X: dataset in the same format of model inputs
        Y: truth spans
        T: taboo words
        texts: a toxic texts
    '''
    X, Y, T, texts = [], [], [], []
    if logger:
        for i in tqdm(range(len(idx))):   
            x, y, taboo_words, text = datamodel.transform(train.iloc[idx[i]])
            X.append(x)
            Y.append(y)
            T.append(taboo_words)
            texts.append(text)
    else:
        for i in range(len(idx)):   
            x, y, taboo_words, text = datamodel.transform(train.iloc[idx[i]])
            X.append(x)
            Y.append(y)
            T.append(taboo_words)
            texts.append(text)
    return X, Y, T, texts

def average(f1_cv, p_cv, r_cv):
    '''
        Averaging Folds 
    Arguments:
        f1_cv(list): list of f1 scores for each folds
        p_cv(list): list of p scores for each folds
        r_cv(list): list of r scores for each folds
    Returns: None
    '''
    f1_cv = np.array(f1_cv)
    r_cv = np.array(r_cv)
    p_cv = np.array(p_cv)
    #print("Cross-validation AVG    ", end='  ')
    print (f"F1 = {f1_cv.mean():.2f} ± {sem(f1_cv):.2f}", end=', \t ')
    print (f"P = {p_cv.mean():.2f} ± {sem(p_cv):.2f}", end=', \t ')
    print (f"R = {r_cv.mean():.2f} ± {sem(r_cv):.2f}")
    

