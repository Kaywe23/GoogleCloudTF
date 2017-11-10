import csv
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
import sys

reload(sys)
sys.setdefaultencoding('latin-1')
from datetime import datetime
import time

lemmatizer = WordNetLemmatizer()
from StringIO import StringIO

# export GOOGLE_APPLICATION_CREDENTIALS='/home/kevin/GoogleCloudTF/train/MachineLearning DC-d672249f7ad8.json'

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 2
datenanzahl = 100
display_step = 1
logs_path = './tmp/Test.py/' + datetime.now().isoformat()
csv_file = '/home/kevin/ml-evaluation (lokal)/train_converted_vermischt.csv'
csv_file2 = '/home/kevin/ml-evaluation (lokal)/vector_test_converted.csv'
checkpoint = '/home/kevin/GoogleCloudTF/train/model.ckpt'
with open('/home/kevin/ml-evaluation (lokal)/lexikon2.pickle', mode='rb') as p:
    lexikon = pickle.load(p)

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
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


saver = tf.train.Saver()
tf_log = 'tf.log'


def trainDNN():
    with tf.name_scope('Model'):
        # Model
        prediction = neural_network_model(x)
    with tf.name_scope('Loss'):
        # Minimize error using cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    with tf.name_scope('AO'):
        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.name_scope('Accuracy'):
        # Accuracy
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    cost_summary = tf.summary.scalar("cost", cost)
    acc_summary = tf.summary.scalar("accuracy", accuracy)
    test_cost_summary = tf.summary.scalar("cost", cost)
    test_acc_summary = tf.summary.scalar("accuracy", accuracy)

    # summary_op = tf.summary.merge_all()



    with tf.Session() as sess:

        sess.run(init)
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        print('Start Training')
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
            print('START:', epoch)
        except:
            epoch = 1
        for epoch in range(hm_epochs):
            if epoch != 1:
                saver.restore(sess, checkpoint)
            avg_cost = 0.
            batch_count = int(datenanzahl)

            with tf.gfile.Open(csv_file, 'rb') as gcs_file:
                lines = gcs_file.readlines()

                zaehler = 0
                for zeile in lines:
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

                    batch_x = np.array([list(features)])
                    batch_y = np.array([eval(label)])

                    _, c, train_cost, train_acc, _cost_summary, _acc_summary = sess.run(
                        [optimizer, cost,cost, accuracy, cost_summary, acc_summary],
                        feed_dict={x: np.array(batch_x), y: np.array(batch_y)})
                    writer.add_summary(_cost_summary, epoch * datenanzahl + zaehler)
                    writer.add_summary(_acc_summary, epoch * datenanzahl + zaehler)
                    avg_cost += c / datenanzahl

                    if zaehler > datenanzahl:
                        print "Batch mit", datenanzahl, "Daten durchlaufen!"
                        break

                    feature_sets = []
                    labels = []
                    with tf.gfile.Open(csv_file2, 'rb') as gc_file:
                        lines2 = gc_file.readlines()
                        zaehler2 = 0
                        for zeile in lines2:
                            try:
                                features = list(eval(zeile.split('::')[0]))
                                label = list(eval(zeile.split('::')[1]))
                                feature_sets.append(features)
                                labels.append(label)
                                test_x = np.array(feature_sets)
                                test_y = np.array(labels)
                                zaehler2 += 1
                                if i % 100 == 0:
                                    # for log on test data:
                                    test_cost, test_acc, _test_cost_summary, _test_acc_summary = \
                                    sess.run([cost, accuracy, test_cost_summary, test_acc_summary],
                                         feed_dict={x: test_x, y_: test_y})
                                    # write log
                                    writer.add_summary(_test_cost_summary, epoch * datenanzahl + zaehler)
                                    writer.add_summary(_test_acc_summary, epoch * datenanzahl + zaehler)

                                    print(
                                    'Epoch {0:3d}, Batch {1:3d} | Train Cost: {2:.2f} | Test Cost: {3:.2f} | Accuracy batch train: {4:.2f} | Accuracy test: {5:.2f}'
                                    .format(epoch, zaehler2, train_cost, test_cost, train_acc, test_acc))
                            except:
                                pass


            saver.save(sess, checkpoint)


            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')



        print'Getestet:', zaehler

        writer.flush()
        print'Accuracy:', accuracy.eval({x: test_x, y: test_y})


trainDNN()