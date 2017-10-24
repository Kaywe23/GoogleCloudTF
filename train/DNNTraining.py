from __future__ import print_function
import os
import tensorflow as tf
import pickle
import sys
import io
from datetime import datetime
import numpy as np
import logging
from tensorflow.python.lib.io import file_io

import argparse


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



nodes_hidden1 = 500
nodes_hidden2 = 500
klassen = 2
batchzahl = 32
datenanzahl = 2000000
epochen = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

aktuelle_epoche = tf.Variable(1)

hidden_layer1 = {'f_fum':nodes_hidden1,'weight':tf.Variable(tf.random_normal([2638, nodes_hidden1])),
                                                                    'bias':tf.Variable(tf.random_normal([nodes_hidden1]))}
hidden_layer2 = {'f_fum':nodes_hidden2,'weight':tf.Variable(tf.random_normal([nodes_hidden1,nodes_hidden2])),
                                                                    'bias':tf.Variable(tf.random_normal([nodes_hidden2]))}
output_layer = {'f_fum':None,'weight':tf.Variable(tf.random_normal([nodes_hidden2, klassen])),
                                                                    'bias':tf.Variable(tf.random_normal([klassen])),}

  
def neural_network(daten):
    l1 = tf.add(tf.matmul(daten,hidden_layer1['weight']), hidden_layer1['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_layer2['weight']), hidden_layer2['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output

saver = tf.train.Saver()
tf_log = 'tf.log'


def trainDNN(x,  pickle_file='train/lexikon.pickle', 
                            job_dir='./tmp/DNNTraining',  **args):
    csv_file_1 = file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/train_converted_vermischt.csv')
    csv_file_2 = file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/vector_test_converted.csv')
    model = file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/model.ckpt')
   # pickle_file='gs://machinelearning-dc-bucket/input/lexikon.pickle'
    
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            epoche = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('START:',epoche)
        except:
            epoche = 1
        while epoche <= epochen:
            if epoche != 1:
                saver.restore(sess, model)
            epoch_loss = 1
            with file_io.FileIO(pickle_file,'r+') as f:
                if sys.version_info<(3, ):
                    lexikon=pickle.load(f)
                else:
                    lexikon = pickle.load(f, encoding='bytes')
            with io.open(csv_file_1,buffering=20000,encoding='latin-1') as f:
                zaehler = 0
                for zeile in f:
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
                    print(batch_x)
                    print(batch_y)
                    _, c = sess.run([optimizer, cost],feed_dict={x:np.array(batch_x), y: np.array(batch_y)})
                    epoch_loss += c
                    if zaehler < datenanzahl:
                        print('Es wurden', datenanzahl, 'daten verarbeitet')
                saver.save(sess, model)
                print('Es sind', epoche, 'Epochen von', epochen, 'fertig,loss:',epoch_loss)
                
                with open(tf_log,'a') as f:
                    f.write(str(epoche)+'\n')
                epoche +=1
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                feature_sets = []
                labels = []
                zaehler = 0
                with open(csv_file_2, buffering=20000) as f:
                    for zeile in f:
                        try:
                            features = list(eval(zeile.split('::')[0]))
                            label = list(eval(zeile.split('::')[1]))
                            feature_sets.append(features)
                            labels.append(label)
                            zaehler += 1
                        except:
                            pass
                print('Getestet:',zaehler)
                test_x = np.array(feature_sets)
                test_y = np.array(labels)
                print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
   '--pickle-file',  help='CSB training pickle')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments=args.__dict__
    trainDNN(x, **arguments)
 


def testDNN():
    csv_file_1 = file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/train_converted_vermischt.csv')
    csv_file_2 = file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/vector_test_converted.csv')
    lexiconfile=  file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/lexikon.pickle')
    model = file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/model.ckpt')
  
    prediction = neural_network(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoche in range(epochen):
            try:
                saver.restore(sess, model)
            except Exception as e:
                print(str(e))
            epoch_loss=0
            
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            feature_sets = []
            labels = []
            zaehler = 0
            with open(csv_file_2, buffering=20000) as f:
                for zeile in f:
                    try:
                        features = list(eval(zeile.split('::')[0]))
                        label = list(eval(zeile.split('::')[1]))
                        #print(features)
                       # print(label)
                        feature_sets.append(features)
                        labels.append(label)
                        zaehler += 1
                    except:
                        pass
            print('Getested:',zaehler)
            test_x = np.array(feature_sets)
            test_y = np.array(labels)
            print('Accuracy:',accuracy.eval({x:test_x, y: test_y}))

#testDNN()

def useDNN(input_data):
    lexiconfile=  file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/lexikon.pickle')
    model = file_io.read_file_to_string('gs://machinelearning-dc-bucket/input/model.ckpt')
    prediction = neural_network(x)
    with open(lexiconfile,'rb') as f:
        lexikon = pickle.load(f)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"model.ckpt")
        woerter = word_tokenize(input_data.lower())
        woerter = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexikon))
        for wort in woerter:
            if wort.lower() in lexikon:
                indexWert = lexikon.index(wort.lower())
                features[indexWert] += 1
                features = np.array(list(features))
                result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1)))
                if result[0] == 0:
                    print('Positive:',input_data)
                elif result[0] == 1:
                    print('Negative:',input_data)
