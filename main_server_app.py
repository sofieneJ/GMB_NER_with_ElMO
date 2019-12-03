# This model predicts NERs amongst {org, per, tim, geo(Geographical Entity), gpe(Geopolitical Entity)}
# The model only accepts email with a number of tokens less than 512 after preprocessing. Extremely long emails will be refused
# @author : sofiene.jenzri@uipath.com
# @version: 1.0 05/10/2019

# cf https://www.kaggle.com/navya098/bi-lstm-for-ner
import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.layers.merge import add
import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


# import pickle
import os
import json

def build_model(max_seq_len, vocab_size, num_tags):
    embedding_dim = 256
    input = Input(shape=(max_seq_len,))
    embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len)(input)
    #     embeddings = Dropout(0.1)(embeddings)
    x1 = Bidirectional(LSTM(units=256, return_sequences=True, recurrent_dropout=0.1))(embeddings)

    x2 = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x1)
    #     x = add([x1, x2])  # residual connection to the first biLSTM

    out = TimeDistributed(Dense(units=num_tags, activation="softmax"))(x2)  # softmax output layer

    model = Model(input, out)
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])#, sample_weight_mode="temporal"

    print (model.summary())

    return model

class NERExtractor():
    def __init__(self):
        
        trained_model_path =  os.path.realpath("./server_models/NER_BiLSTM_keras.h5")
        vocab_dic_path = os.path.realpath('./server_models/vocab_dic.json')
        tags_dic_path = os.path.realpath('./server_models/tags_dic.json')
        model_config_path = os.path.realpath('./server_models/model_config.json')        

        with open(vocab_dic_path, 'r') as f:
            self.word2idx = json.load(f)
        with open(tags_dic_path, 'r') as f:
            self.tags2idx=json.load(f)
        with open(model_config_path, 'r') as f:
            self.config_dic=json.load(f)

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            with self.session.as_default():
                self.model = build_model(max_seq_len=self.config_dic["max_seq_len"], vocab_size=len(self.word2idx), num_tags=len(self.tags2idx))
                self.model.load_weights(trained_model_path)
                                        

        self.idx2word = {idx:word for word,idx in self.word2idx.items()}
        self.idx2tag = {idx:tag for tag,idx in self.tags2idx.items()} 


    def predict(self, input):
        text_sequence = input.split()
        X_test = [[self.word2idx[w] for w in text_sequence]]
        # padding
        X_test = pad_sequences(maxlen=self.config_dic["max_seq_len"], sequences=X_test, padding="post",value=self.word2idx[self.config_dic["PAD_WORD"]])
        # print (X_test)

        with self.graph.as_default():
            with self.session.as_default():
                p = self.model.predict(X_test)
        p = np.argmax(p, axis=-1)

        predicted_tags_seq = [self.idx2tag[pred]  for w, pred in zip(X_test[0], p[0])  if self.idx2word[w]!='ENDPAD']
        # print (predicted_tags_seq)
        print(input)
        predicted_entities = []
        for i,entity_name in enumerate(predicted_tags_seq):
            if entity_name != self.config_dic["PAD_TAG"] and entity_name[0]=="B":
                start_index = len(' '.join(text_sequence[:i]))
                j = i+1
                entity_len = len(text_sequence[i])
                entity_text = text_sequence[i]
                while (j<len(predicted_tags_seq) and predicted_tags_seq[j][2:]==entity_name[2:] and predicted_tags_seq[j][0]=="I"):
                    entity_len += len(text_sequence[j])+1
                    entity_text += str (' '+text_sequence[j])
                    j+=1

                entity = {
                    "entity":entity_name,
                    "entity_text":entity_text,
                    "start_index":start_index,
                    "entity_length":entity_len
                }
                predicted_entities.append(entity)

        print (json.dumps(predicted_entities, sort_keys=True, indent=2) )
        return json.dumps(predicted_entities, sort_keys=True, indent=2)


def test_main():

    test_corpus_path = './temp_data/test_copus.txt'
    with open(test_corpus_path, "r", encoding = "utf8") as f:
        test_texts = f.readlines()
        print (f'test corpus contains {len(test_texts)} sequences')
    # with open(test_tags_path, "r", encoding = "utf8") as f:
    #     test_tags = f.readlines()
    test_texts = sorted(test_texts, key=lambda text:len(text), reverse=True)
    my_classifier = NERExtractor()
    for i in range (20,25):
        my_classifier.predict(test_texts[i])#, test_tags_path.split()



if __name__=='__main__':
    test_main()
