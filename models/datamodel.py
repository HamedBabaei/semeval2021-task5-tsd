import spacy
nlp = spacy.load('en_core_web_sm')
from models.fix_spans import fix_spans
import itertools
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('english')) 

class DataModel:
    def __init__(self, model='crf', remove_stop_words=False, split_toxics=False):
        '''
            Initialize Data Models
        Arguments: 
            model(string): name of data model method to call
        Returns: None
        '''
        self.__func = lambda x:''.join([' '+x[0][x[1][i]] if x[1][i-1] + 1 != x[1][i] else x[0][x[1][i]] 
                                      for i in range(len(x[1]))])
        self.__model = model
        self.__stop_word_removal = remove_stop_words
        self.__split_toxics = split_toxics
        self.__func_seprate_toxics = lambda x:[taboo for taboo in ''.join(['<TABOO>'+x[0][x[1][i]] if x[1][i-1] + 1 != x[1][i] else x[0][x[1][i]] 
                                    for i in range(len(x[1]))]).split('<TABOO>') if len(taboo) != 0]
    
    def transform(self, X):
        '''
            Transforming Data into models inputs
        Arguments:
            X(DataFrame): a dataset in pandas dataframe, (a sample only)
        Returns:
            transformed_data
        '''
        if self.__model == 'crf':
            return self.__crf_transform(X)
        elif self.__model == 'random':
            return self.__random_transform(X)
        elif self.__model == 'hmm':
            return self.__hmm_transform(X)
        elif self.__model == 'cmm':
            return self.__cmm_transform(X)
        return None
    
    def __crf_transform(self, X):
        '''
            Transforming Data into CRF model format
        Arguments:
            X(DataFrame): a dataset in pandas dataframe, (a sample only)
        Returns:
            X: dataset format of [(w1, pos1, label),....]
            spans(string): spans
            toxic_words(list): taboo words
            text(string): text
        '''
        text, spans = X['text'], X['spans']
        spans = fix_spans(spans, text)
        toxic_words = self.__func([text, spans])
        nlp_text = nlp(text)
        cns_ranges = self.__contiguous_ranges(spans)
        X , txt = [], ""
        indecis = [(m.start(), m.end()-1) for m in re.finditer(r'\S+', text)]
        for indec in indecis:
            span = [i for i in range(indec[0], indec[1]+1)]
            ws = text[indec[0]: indec[1]+1]
            exist = False
            for _word in nlp_text:
                if ws == _word.text:
                    words, exist = [_word], True
                    break
            if not exist:
                words = nlp(ws)
            for word in words:                
                if (word.text.lower() in stop_words) and self.__stop_word_removal:
                    continue
                if self.__word_validation(word):
                    continue
                span = list(range(indec[0] + ws.index(word.text), indec[0] + ws.index(word.text)+len(word.text)))
                if word.text in toxic_words and self.__check_toxicity(word.text, len(txt), text, cns_ranges):
                    X.append((word.text, word.tag_,'toxic', span))
                else:
                    X.append((word.text, word.tag_, 'normal', span))
            txt += ws + ' '
        if self.__split_toxics:
            toxic_words = self.__func_seprate_toxics([text, spans])
        return X, spans, toxic_words, text
    
    def __word_validation(self, word):
        if word.pos_ == 'SPACE':
            return True 
        if word.tag_ in ['.', ',', "''", "``", 'CD', ":","XX", "NFP"]:
            return True
        if word.text in ['-', "n't", "%", ":", '.', ',', "''", "``", '"', "'s", "=", "-", "'re","'ve","'m","'ll"]:
            return True
        if word.text in ["'","..", "wo","ca","nt", '', ')', '(']:
            return True
        return False

    def __cmm_transform(self, X):
        '''
            Transforming Data into CRF model format
        Arguments:
            X(DataFrame): a dataset in pandas dataframe, (a sample only)
        Returns:
            X: dataset format of [(w1, pos1, label),....]
            spans(string): spans
            toxic_words(list): taboo words
            text(string): text
        '''
        text, spans = X['text'].replace('\n',' '), X['spans']
        spans = fix_spans(spans, text)
        toxic_words = self.__func([text, spans])
        nlp_text = nlp(text)
        cns_ranges = self.__contiguous_ranges(spans)
        X , txt = [], ""
        indecis = [(m.start(), m.end()-1) for m in re.finditer(r'\S+', text)]
        for indec in indecis:
            span = [i for i in range(indec[0], indec[1]+1)]
            ws = text[indec[0]: indec[1]+1]
            exist = False
            for _word in nlp_text:
                if ws == _word.text:
                    words, exist = [_word], True
                    break
            if not exist:
                words = nlp(ws)
            for word in words:         
                if (word.text.lower() in stop_words) and self.__stop_word_removal:
                    continue
                if self.__word_validation(word):
                    continue
                span = list(range(indec[0] + ws.index(word.text), indec[0] + ws.index(word.text)+len(word.text)))
                if word.text in toxic_words and self.__check_toxicity(word.text, len(txt), text, cns_ranges):
                    X.append((word.text, word.tag_, word.ent_iob_,'toxic', span))
                else:
                    X.append((word.text, word.tag_, word.ent_iob_, 'normal', span))
            txt += ws + ' '
        if self.__split_toxics:
            toxic_words = self.__func_seprate_toxics([text, spans])
        return X, spans, toxic_words, text
    
    def __check_toxicity(self, word, start, text, ent_ranges):
        index = text.index(word, start)
        for ent_range in ent_ranges:
            if index >= ent_range[0] and index <= ent_range[1]:
                return True
        return False

    def __contiguous_ranges(self, span_list):
        """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)]."""
        output = []
        for _, span in itertools.groupby(
            enumerate(span_list), lambda p: p[1] - p[0]):
            span = list(span)
            output.append((span[0][1], span[-1][1]))
        return output
    
    def __hmm_transform(self, X):
        X, spans, toxic_words, text = self.__crf_transform(X)
        X = [(x[0], x[2]) for x in X]
        return X, spans, toxic_words, text

    def __random_transform(self, X):
        '''
            Transforming Data into CRF model format
        Arguments:
            X(DataFrame): a dataset in pandas dataframe, (a sample only)
        Returns:
            text(string): text
            spans(string): spans
            None: to keep interface the same!
            text(string): text
        '''
        text, spans = X['text'], X['spans']
        spans = fix_spans(spans, text)
        return text, spans, None, text
    