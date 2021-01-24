import nltk
from nltk.probability import LidstoneProbDist

class HMM:

    def __init__(self):
        self.char_set = set()

    def fit(self, X, Y=None):
        self.__init__()
        #Y = [label for x in X for word, label in x]
        words = [word for x in X for word, label in x]
        #for word in X:
        #    for char in word:
        #        self.char_set.add(char)
        #self.entity_set = set(Y)
        #trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=self.entity_set, symbols=self.char_set)
        estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins)
        trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(states=list(range(10)))
        self.model = trainer.train(X, estimator=estimator)

    def predict(self, X):
        predicts = []
        for x in X:
            words = [word for word, label in x]
            preds = [pred for word, pred in self.model.tag(words)]
            predicts.append(preds)
        return predicts
