
from metrics import evaluate
import argparse
import pandas as pd
from ast import literal_eval

def get_args():
    '''
        Get truth and predicts path from user
    Arguments: None
    Returns:
        args: argparse which contains args.t path to the grand truths df and 
                                      args.p path to the predictions df
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--truth', help="path to the grand truths")
    parser.add_argument('-p', '--predicts', help="path to the predictions")
    args = parser.parse_args()
    if args.truth is None or args.predicts is None:
        parser.print_usage()
        exit()
    return args

def main():
    args = get_args()
    truth = pd.read_csv(args.truth)
    predicts = pd.read_csv(args.predicts)
    truth['spans'] = truth.spans.apply(literal_eval)
    predicts['spans'] = predicts.spans.apply(literal_eval)

    predictions = predicts['spans'].tolist()
    gold = truth['spans'].tolist()
    f1, p, r = evaluate(gold, predictions)
    print("F1-Score :", f1 )
    print("Precision:", p)
    print("Recall   :", r)

main()