from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, make_scorer
import pickle
import pandas as pd
from processor import preprocessing
from processor import create_tfidf
import nltk

def main():
    train = pd.read_csv('resources/train.csv')
    target = train['Is_Response']
    del train["Is_Response"]
    target = [1 if x == 'happy' else 0 for x in target]
    nltk.download('stopwords')
    train = preprocessing(train)
    train = create_tfidf(train)
    trainAndSaveModel(train,target)

def trainAndSaveModel(train, target):
    mod = GaussianNB()
    print(cross_val_score(mod, train, target, cv=5, scoring=make_scorer(accuracy_score)))
    mod.fit(train, target)
    filename = 'model/finalized_model.sav'
    pickle.dump(mod, open(filename, 'wb'))


if __name__ == '__main__':
    main()
