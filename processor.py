import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import pickle
import config



def preprocessing(df):
    df['word_count'] = df['Description'].apply(lambda x: len(str(x).split(" ")))
    df['char_count'] = df['Description'].str.len()
    df['avg_word'] =  df['Description'].apply(lambda x: avg_word(x))
    stop = stopwords.words('english')

    df['stopwords'] = df['Description'].apply(lambda x: len([x for x in x.split() if x in stop]))
    df['upper'] = df['Description'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    df['Description'] = df['Description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    df['Description'] = df['Description'].str.replace('[^\w\s]', '')
    stop = stopwords.words('english')

    df['Description'] = df['Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['Description'] = df['Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    df['sentiment'] = df['Description'].apply(lambda x: TextBlob(x).sentiment[0])
    df = encodeCategorical(df)
    return df


def encodeCategorical(df):
    cols = ['Browser_Used', 'Device_Used']

    for x in cols:
        lbl = LabelEncoder()
        df[x] = lbl.fit_transform(df[x])
    return df

def create_tfidf(df):
    tfidfvec = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=150, max_features=500)
    tfidfdata = tfidfvec.fit_transform(df['Description'])
    tfidf_df = pd.DataFrame(tfidfdata.todense())
    tfidf_df.columns = ['col' + str(x) for x in tfidf_df.columns]
    del df['Description']
    del df['User_ID']
    return pd.concat([df, tfidf_df], axis=1)

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

def to_labels(x):
    if x == 1:
        return "happy"
    return "not_happy"

def main():
    test = pd.read_csv(config.test_file_path)
    filename = 'model/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    sub = pd.DataFrame({'User_ID': test['User_ID']})
    test = preprocessing(test)
    test = create_tfidf(test)

    sub['Is_Response'] = loaded_model.predict(test)
    sub['Is_Response'] = sub['Is_Response'].map(lambda x: to_labels(x))
    sub = sub[['User_ID', 'Is_Response']]
    sub.to_csv(config.output_file_path, index=False)


if __name__ == '__main__':
    main()


