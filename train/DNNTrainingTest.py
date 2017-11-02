import tensorflow as tf
import pickle
import numpy as np
from tensorflow.python.lib.io import file_io
import argparse
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import io
import csv
import pandas as pd

lemmatizer = WordNetLemmatizer()
from StringIO import StringIO
#export GOOGLE_APPLICATION_CREDENTIALS='/home/kevin/GoogleCloudTF/train/MachineLearning DC-d672249f7ad8.json'

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10
datenanzahl = 2000000




def train_neural_network(train_file='lexikon.pickle',csv_file='train_converted_vermischt.csv',
                         csv_file2='vector_test_converted.csv', job_dir='./tmp/DNNTrainingLite',**args):
    file_stream = file_io.FileIO(train_file, mode='r')
    lexikon = pickle.load(file_stream)



    x = tf.placeholder('float')
    y = tf.placeholder('float')

    hidden_1_layer = {'f_fum': n_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([2578, n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum': n_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'f_fum': n_nodes_hl3,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }

    # Nothing changes
    def neural_network_model(data):
        l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

        return output

    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels= y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        epoch=1
        while epoch <= hm_epochs:

            epoch_loss = 1
            gcs_file = tf.gfile.Open(csv_file, 'rb')
            lines=gcs_file.readlines()
            #csv_reader=csv.reader(gcs_file)

            #csv_reader=gcs_file.read()
            #with io.open(csv_file1, buffering=20000, encoding='latin-1') as f:
            #df_train = pd.read_csv(tf.gfile.Open("./train.csv"),skipinitialspace=True)
            zaehler = 0
            for zeile in lines:
                label = zeile.split(':::')[0]
                tweet = zeile.split(':::')[1]
                woerter = word_tokenize(tweet.lower())
                woerter = [lemmatizer.lemmatize(i) for i in woerter]
                features = np.zeros(len(lexikon))
                for wort in woerter:
                    if wort.lower() in lexikon:
                        indexWert = lexikon.index(wort.lower())
                        features[indexWert] += 1
                batch_x = np.array([list(features)])
                batch_y = np.array([eval(label)])

                _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x), y: np.array(batch_y)})
                epoch_loss += c
                if zaehler < datenanzahl:
                    print('Es wurden', datenanzahl, 'daten verarbeitet')

            print('Es sind', epoch, 'Epochen von', hm_epochs, 'fertig,loss:', epoch_loss)

            epoch += 1
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            feature_sets = []
            labels = []
            zaehler = 0
            gcs_file = tf.gfile.Open(csv_file2, 'rb')
            lines = gcs_file.readlines()

            for zeile in lines:
                try:
                    features = list(eval(zeile.split('::')[0]))
                    label = list(eval(zeile.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    zaehler += 1
                except:
                    pass
            print('Getestet:', zaehler)
            test_x = np.array(feature_sets)
            test_y = np.array(labels)
            print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-file',
        help='GCS or local paths to training data',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument('--csv-file',
                        help='csv file',
                        required =True)
    parser.add_argument('--csv-file2',
                        help='csv file2',
                        required=True)
    args = parser.parse_args()
    arguments = args.__dict__

    train_neural_network(**arguments)