
def crf_output(test, texts, predicts):
    '''
        Preparing CRF output for evaluation
    Arguments:
        test: a test set in crf input format
        texts: toxic dataset test text
        predicts: prediction of crf
    Returns:
        all_prediction: in the same format of TSD taskts
    '''
    all_predictions = []
    for i, preds in enumerate(predicts):
        predictions = []
        for j, pred in enumerate(preds):
            if pred == 'toxic':
                indecies = test[i][j][-1]
                predictions += indecies
        all_predictions.append(predictions)
    return all_predictions

def random_output(test, texts, predicts):
    '''
        Preparing Random Model output for evaluation, 
        it is only to keep interface the same in cross validations for this Model
    Arguments:
        test: a test set in crf input format
        texts: toxic dataset test text
        predicts: prediction of crf
    Returns:
        predicts: in the same format of TSD taskts
    '''
    return predicts

