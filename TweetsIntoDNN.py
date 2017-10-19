from kafka import KafkaConsumer
import json
from pprint import pprint
import io
import TwitterDataInKNN

from collections import OrderedDict
import time

from kafka import SimpleProducer, SimpleClient

consumer = KafkaConsumer('test',\
	                     group_id=None,\
                         bootstrap_servers=['localhost:9092'],
                         fetch_max_bytes=157286400,
                         auto_offset_reset='earliest',
						 enable_auto_commit=False)
kafka = SimpleClient('localhost:9092')
producer = SimpleProducer(kafka)

for msg in consumer:
    tweet = msg.value.decode('utf-8', 'ignore')
    if "great" in tweet and len(tweet) <50:
            ergebnis = TwitterDataInKNN.use_neural_network(tweet)
            print(ergebnis)
         
        
        





