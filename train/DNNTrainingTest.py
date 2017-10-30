import tensorflow as tf
import pickle
import numpy as np
from tensorflow.python.lib.io import file_io
import argparse
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import io
lemmatizer = WordNetLemmatizer()
from google.cloud import storage

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100
hm_epochs = 10
datenanzahl = 2000000




def train_neural_network(train_file='lexikon.pickle',
                         job_dir='./tmp/DNNTrainingLite',**args):
    file_stream = file_io.FileIO(train_file, mode='r')
    lexikon = pickle.load(file_stream)

    client = storage.Client()
    bucket = client.get_bucket('machinelearning-dc-bucket')
    blob = storage.Blob('train_converted_vermischt.csv', bucket)
    content = blob.download_as_string()


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

            #with io.open(csv_file1, buffering=20000, encoding='latin-1') as f:
                zaehler = 0
                for zeile in content:
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
    #parser.add_argument('--csv-file1',
                       # help='csv file',
                       # required =True)
    args = parser.parse_args()
    arguments = args.__dict__

    train_neural_network(**arguments)