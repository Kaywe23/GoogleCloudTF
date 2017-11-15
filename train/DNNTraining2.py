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

lemmatizer = WordNetLemmatizer()
#flags = tf.app.flags

#flags.DEFINE_integer('hm_epochs',5,'Anzahl epochen')
#FLAGS = flags.FLAGS
n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10
datenanzahl = 10000
display_step = 1

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



def trainDNN(train_file='lexikon2.pickle',csv_file='train_converted_vermischt.csv',
                         csv_file2='vector_test_converted.csv', job_dir='./tmp/DNNTraining2',
                         checkpoint='model.ckpt',logs='tf.log',**args):

    file_stream = file_io.FileIO(train_file, mode='r')
    lexikon = pickle.load(file_stream)

    with tf.name_scope('Model'):
        # Model
        prediction = neural_network_model(x)
    with tf.name_scope('Loss'):
        # Minimize error using cross entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    with tf.name_scope('AO'):
        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    with tf.name_scope('Accuracy'):
        # Accuracy
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    init = tf.global_variables_initializer()

    cost_summary = tf.summary.scalar("cost", cost)
    acc_summary = tf.summary.scalar("accuracy", accuracy)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(job_dir, graph=tf.get_default_graph())
        print('Start Training')
        try:
            epoch = int(tf.gfile.Open(logs,'r').read().split('\n')[-2])+1
            print('START:',epoch)
        except:

            epoch = 1
        for epoch in range(hm_epochs):
            if epoch != 1:
                saver.restore(sess,checkpoint)
            avg_cost=0.
            batch_count=int(datenanzahl)

            with tf.gfile.Open(csv_file, 'rb') as gcs_file:
                lines=gcs_file.readlines()

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

                    _, c = sess.run([optimizer, cost],
                                             feed_dict={x: np.array(batch_x), y: np.array(batch_y)})

                    if zaehler % 100 == 0:
                        summary_str = sess.run(summary_op,
                                               feed_dict={x: batch_x,y: batch_y})
                        summary_writer.add_summary(summary_str, epoch*datenanzahl + zaehler)
                        summary_writer.flush()

                    #writer.add_summary(summary, epoch * datenanzahl + zaehler)

                    avg_cost += c / datenanzahl

                    if zaehler > datenanzahl:
                        print "Batch mit", datenanzahl, "Daten durchlaufen!"
                        break
            saver.save(sess, checkpoint)
            if epoch % display_step == 0:

                print "Epoch:", '%04d' % (epoch+1),"of",'%04d' % (hm_epochs), "cost=", "{:.9f}".format(avg_cost)




            with tf.gfile.Open(logs, 'a') as f:
                f.write(str(epoch) + '\n')

        feature_sets = []
        labels = []
        zaehler = 0

        with tf.gfile.Open(csv_file2, 'rb') as gc_file:
            lines2 = gc_file.readlines()

            for zeile in lines2:
                try:
                    features = list(eval(zeile.split('::')[0]))
                    label = list(eval(zeile.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    zaehler += 1
                except:
                    pass


        print'Getestet:', zaehler
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print'Accuracy:', accuracy.eval({x: test_x, y: test_y})


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
    parser.add_argument('--checkpoint',
                        help='checkpointfile',
                        required=True)
    parser.add_argument('--logs',
                        help='logfile',
                        required=True)
    args = parser.parse_args()
    arguments = args.__dict__

    trainDNN(**arguments)
