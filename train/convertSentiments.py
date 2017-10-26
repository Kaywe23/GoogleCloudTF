import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd
import io

lemmatizer = WordNetLemmatizer()


def convert(infile, outfile):
    out = io.open(outfile, 'a')
    with io.open(infile, buffering=200000, encoding='latin-1') as f:
        try:
            for zeile in f:
                zeile = zeile.replace('"', '')
                initial_polarity = zeile.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1, 0]
                elif initial_polarity == '4':
                    initial_polarity = [0, 1]

                tweet = zeile.split(',')[-1]
                outzeile = str(initial_polarity) + ':::' + tweet
                out.write(outzeile)
        except Exception as e:
            print(str(e))
    out.close()


#convert('training.1600000.processed.noemoticon.csv','train_converted.csv')
#convert('testdata.manual.2009.06.14.csv','test_converted.csv')


def getLexikon(infile):
    lexikon = []
    with io.open(infile, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            zaehler = 1
            content = ''
            for zeile in f:
                zaehler += 1
                if (zaehler / 2500.0).is_integer():
                    tweet = zeile.split(':::')[1]
                    content += ' ' + tweet
                    woerter = word_tokenize(content)
                    woerter = [lemmatizer.lemmatize(i) for i in woerter]
                    lexikon = list(set(lexikon + woerter))
                    print(zaehler, len(lexikon))

        except Exception as e:
            print(str(e))

    with io.open('lexikon.pickle', 'wb') as f:
        pickle.dump(lexikon, f)


#getLexikon('train_converted.csv')


def getVector(infile, outfile, lexikonpickle):
    with io.open(lexikonpickle, 'rb') as f:
        lexikon = pickle.load(f)
    out = open(outfile, 'a')
    with io.open(infile, buffering=20000, encoding='latin-1') as f:
        zaehler = 0
        for zeile in f:
            zaehler += 1
            label = zeile.split(':::')[0]
            tweet = zeile.split(':::')[1]
            woerter = word_tokenize(tweet.lower())
            woerter = [lemmatizer.lemmatize(i) for i in woerter]

            features = np.zeros(len(lexikon))

            for wort in woerter:
                if wort.lower() in lexikon:
                    indexWert = lexikon.index(wort.lower())

                    features[indexWert] += 1

            features = list(features)
            outzeile = str(features) + '::' + str(label) + '\n'
            out.write(outzeile)

        print(zaehler)


#getVector('test_converted.csv', 'vector_test_converted.csv', 'lexikon.pickle')


def datenVermischen(infile):
    d = pd.read_csv(infile, error_bad_lines=False)
    d = d.iloc[np.random.permutation(len(d))]
    print(d.head())
    d.to_csv('train_converted_vermischt.csv', index=False)


#datenVermischen('train_converted.csv')


def testdatenAufbereiten(infile):
    feature_sets = []
    labels = []
    zaehler = 0
    with io.open(infile, buffering=20000) as f:
        for zeile in f:
            try:
                features = list(eval(zeile.split('::')[0]))
                label = list(eval(zeile.split('::')[1]))

                feature_sets.append(features)
                labels.append(label)
                zaehler += 1
            except:
                pass
    print(zaehler)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)


#testdatenAufbereiten('vector_test_converted.csv')


f=open('lexikon.pickle', 'r')
lexikon=pickle.load(f)
print(len(lexikon))
