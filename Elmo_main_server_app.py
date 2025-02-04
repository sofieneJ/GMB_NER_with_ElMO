# This model predicts NERs amongst {org, per, tim, geo(Geographical Entity), gpe(Geopolitical Entity)}
# This work is based on Tobias Sterbak' work https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/
# @author : sofiene.jenzri@uipath.com
# @version: 1.0 05/12/2019

# cf https://www.kaggle.com/navya098/bi-lstm-for-ner
from helpers import *
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras.optimizers import Adam
from keras.layers.merge import add
import tensorflow as tf
import tensorflow_hub as hub
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


# import pickle
import os
from time import time

from nltk.tokenize import sent_tokenize


def build_model(pretrained_ElMO_module, max_seq_len, num_tags, batch_size, learning_rate):    
 
    def ElmoEmbedding(x):
        return pretrained_ElMO_module(inputs={
                                "tokens": tf.squeeze(tf.cast(x, tf.string)),
                                "sequence_len": tf.constant(batch_size*[max_seq_len])
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]
    
    embedding_dim = 1024
    input_text = Input(shape=(max_seq_len,), dtype=np.string_)#, dtype=tf.string
    embedding = Lambda(ElmoEmbedding, output_shape=( max_seq_len, embedding_dim))(input_text)
    
    nb_units = 128
    x1 = Bidirectional(LSTM(units=256, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)

    x2 = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x1)
#     x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(num_tags, activation="softmax"))(x2)
    model = Model(input_text, out)
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    print (model.summary())
    return model

def pad_word_sequence (Corpus, max_seq_len, pad_word):
  padded_corpus = []
  for seq in Corpus:
    new_seq = []
    for i in range(max_seq_len):
      if i < len(seq):
        new_seq.append(seq[i])
      else:
        new_seq.append(pad_word)
    padded_corpus.append(new_seq)
  return padded_corpus

class NERExtractor():
    def __init__(self, appName):
        
        pretrained_ElMO_path =  os.path.realpath("../pretrained_models/hub_elmo/")

        trained_model_path =  f"./models/ElMO/{appName}/ElMo_BiLSTM_keras.h5"
        retrained_model_path = f"./models/ElMO/{appName}/retrained_ElMo_BiLSTM_keras.h5"
        archive_retrained_model_path = f"./models/ElMO/{appName}/archive_retrained_ElMo_BiLSTM_keras.h5"
        self.retrained_model_path =  os.path.realpath(retrained_model_path)
        self.archive_retrained_model_path =  os.path.realpath(archive_retrained_model_path)
        
        tags_dic_path = os.path.realpath(f"./models/ElMO/{appName}/tags_dic.json")
        model_config_path = os.path.realpath(f"./models/ElMO/{appName}/model_config.json")        
        taxonomy_map_path = os.path.realpath(f"./models/ElMO/{appName}/taxonomy_mapping.json")

        self.retraining_text_seq_path = os.path.realpath(f"./new_training_data/{appName}/hot_folder/text_sequences.txt")
        self.retraining_tags_seq_path = os.path.realpath(f"./new_training_data/{appName}/hot_folder/tags_sequences.txt")

        self.retraining_archive_text_seq_path = os.path.realpath(f"./new_training_data/{appName}/archived_data/text_sequences.txt")
        self.retraining_archive_tags_seq_path = os.path.realpath(f"./new_training_data/{appName}/archived_data/tags_sequences.txt")

        with open(tags_dic_path, 'r') as f:
            self.tags2idx=json.load(f)
        with open(model_config_path, 'r') as f:
            self.config_dic=json.load(f)
        with open(taxonomy_map_path, 'r') as f:
            self.taxo_map = json.load(f)

        self.appName=appName
        self.session = tf.compat.v1.Session()
        self.graph = tf.compat.v1.get_default_graph()

        class myCallback(tf.keras.callbacks.Callback):
            def __init__(self, acc_thres, **kwargs):
                self.acc_thres = acc_thres
                super(myCallback, self).__init__(**kwargs)

            def on_epoch_end(self, epoch, logs={}):
                if (logs.get('val_accuracy')>self.acc_thres):
                    print(f"Reached {self.acc_thres} accuracy so stopping the training!")
                    self.model.stop_training = True


        with self.graph.as_default():
            with self.session.as_default():
                self.accuracy_training_callback = myCallback(acc_thres = self.config_dic["retraining_acc_thres"])
                pretrained_elmo_model = hub.Module(spec=pretrained_ElMO_path, trainable=True)
                self.model = build_model(pretrained_ElMO_module=pretrained_elmo_model, max_seq_len=self.config_dic["max_seq_len"], 
                                        num_tags=len(self.tags2idx), batch_size=self.config_dic["batch_size"], learning_rate=self.config_dic["learning_rate"])
                if (self.config_dic["initial_weights"]=="retrained" and os.path.exists(retrained_model_path)):
                    self.model.load_weights(os.path.realpath(retrained_model_path))
                    print (f"INFO: I loaded retrained model from {retrained_model_path}")
                elif (self.config_dic["initial_weights"]=="pretrained" and os.path.exists(trained_model_path)):
                    self.model.load_weights(os.path.realpath(trained_model_path))
                    print (f"INFO: I loaded pretrained model from {trained_model_path}")
                elif (self.config_dic["initial_weights"]=="archive" and os.path.exists(archive_retrained_model_path)):
                    self.model.load_weights(os.path.realpath(archive_retrained_model_path))
                    print (f"INFO: I loaded archive-trained model from {archive_retrained_model_path}")
                else:
                    print (f"INFO: I did not load any pretrained weights")
                    
                                        
        self.idx2tag = {idx:tag for tag,idx in self.tags2idx.items()} 


    def predict(self, input):
        #Load model params
        max_seq_len = self.config_dic["max_seq_len"]
        PAD_WORD = self.config_dic['PAD_WORD']
        batch_size = self.config_dic['batch_size']
        min_seq_len = self.config_dic["min_seq_len"]  
        
        sents = sent_tokenize(input)
        sents = [sent for sent in sents if len(sent.split())<max_seq_len and len(sent.split())>min_seq_len]
        nb_sentences = len(sents)
        assert (nb_sentences>0)


        np_sequences = np.empty((0,max_seq_len))
        np_predictions = np.empty((0,max_seq_len))
        np_probas = np.empty((0,max_seq_len))

        for i in range (0, nb_sentences//batch_size if (nb_sentences%batch_size == 0) else nb_sentences//batch_size+1):
            if nb_sentences>(i+1)*batch_size:
                sents_batch = sents[i*batch_size:(i+1)*batch_size]
            else:
                sents_batch = sents[i*batch_size:]

            sequences_list = [sent.split() for sent in sents_batch]
            batch_np_sequences = np.array(pad_word_sequence(sequences_list, max_seq_len, PAD_WORD))
            # padding the batch to reach batch_size
            if len(sents_batch) < batch_size:
                batch_pad = np.array([[PAD_WORD for _ in range (0, max_seq_len)] for _ in range(0,batch_size-len(sents_batch))])
                batch_np_sequences = np.vstack((batch_np_sequences,batch_pad))
            t0 = time()
            with self.graph.as_default():
                with self.session.as_default():
                    p = self.model.predict(batch_np_sequences, batch_size=batch_size)
            print(f'It took the model {time()- t0} seconds to infer the extraction')
            batch_predictions = np.argmax(p, axis=-1)
            batch_probas = np.max(p, axis=-1)

            #appending batch predictions
            np_sequences = np.vstack((np_sequences, batch_np_sequences))
            np_predictions = np.vstack((np_predictions, batch_predictions))
            np_probas = np.vstack((np_probas, batch_probas))

        predicted_entities = []
        for k, sent in enumerate(sents):
            predicted_tags_seq = [self.idx2tag[pred]  for w, pred in zip(np_sequences[k], np_predictions[k])  if w!= PAD_WORD]
            # print (predicted_tags_seq)
            # print(np_sequences[k])
            for i,entity_name in enumerate(predicted_tags_seq):
                if entity_name != self.config_dic["PAD_TAG"] and entity_name[0]=="B":
                    start_index = sent.index(np_sequences[k][i], len(' '.join(np_sequences[k][:i]))) + input.index(sent, sum([len(sentence) for h,sentence in enumerate(sents) if h<k]))
                    j = i+1
                    entity_len = len(np_sequences[k][i])
                    entity_text = np_sequences[k][i]
                    count = 1
                    proba = np_probas[k][i]
                    while (j<len(predicted_tags_seq) and predicted_tags_seq[j][2:]==entity_name[2:] and predicted_tags_seq[j][0]=="I"):
                        entity_len += len(np_sequences[k][j])+1
                        entity_text += str (' '+np_sequences[k][j])
                        proba += np_probas[k][j]
                        count+=1
                        j+=1
                    proba = (proba/count)

                    entity = {
                        "entity":entity_name,
                        "entity_text":entity_text.replace("\"","'"),
                        "start_index":start_index,
                        "entity_length":entity_len,
                        "proba":proba
                    }
                    predicted_entities.append(entity)

        print (json.dumps(predicted_entities, sort_keys=True, indent=2))
        return json.dumps(predicted_entities, sort_keys=True, indent=2)

    def retrain (self, text, validation_results):

        json_extraction = json.loads(validation_results)
        tags_list =[]
        for entity_group in json_extraction["ResultsDocument"]["Fields"]:
            for entity in entity_group["Values"]:
                entity_text = entity["Value"]
                start_index = entity["Reference"]["TextStartIndex"]
                entity_text_length =  entity["Reference"]["TextLength"]
                tag = self.taxo_map [entity_group["FieldId"]]
                token_start_index = start_index
                counter = 0
                for i, token in enumerate(entity_text.split()):
                    token_start_index = start_index + entity_text.index(token, counter)
                    token_tag = "B"+tag if i==0 else "I"+tag 
                    counter += len(token)
                    my_tag= {
                        "token":token,
                        "start_index":token_start_index,
                        "tag":token_tag
                    }
                    tags_list.append(my_tag)
        
        max_seq_len = self.config_dic["max_seq_len"]  
        min_seq_len = self.config_dic["min_seq_len"]  
        
        sents = sent_tokenize(text)
        sents = [sent for sent in sents if len(sent.split())<max_seq_len and len(sent.split())>min_seq_len]
        nb_sentences = len(sents)

        nb_hits = 0
        for i,sent in enumerate(sents):
            tag_sequence= []
            for j,token in enumerate(sent.split()):
                index_in_text = text.index(sent, sum([len(s) for k,s in enumerate(sents) if k<i])) + sent.index(token, sum([(len(t)+1) for k,t in enumerate(sent.split()) if k<j ]))
                hits = list(filter(lambda tag: tag["start_index"]==index_in_text, tags_list))
                if len(hits) != 0:
                    assert(len(hits)==1)
                    assert (hits[0]["token"]==token or hits[0]["token"].replace("'", "\"") == token)
                    tag_sequence.append(hits[0]["tag"])
                    nb_hits+=1
                else:
                    tag_sequence.append("O")
             
            with open(self.retraining_text_seq_path,"a", encoding="utf-8") as f:
                f.write(sent.replace('\n',' '))
                f.write("\n")
            with open(self.retraining_tags_seq_path,"a") as f:
                f.write(','.join(tag_sequence))
                f.write("\n")
        assert (len(tags_list)==nb_hits)

        #check if the batch threshold is reached
        batch_threshold = self.config_dic["retraining_batch_threshold"]
        with open(self.retraining_tags_seq_path,"r") as f:
            if len(f.readlines()) >= batch_threshold:
                return self.fire_retraining()
            else:
                return "retraining sample stored for next retraining when the threshold is reached"

                
    def fire_retraining (self):
        with open(self.retraining_text_seq_path,"r", encoding="utf-8") as f:
            corpus = [sent.split() for sent in f.readlines()]
        with open(self.retraining_tags_seq_path,"r") as f:
            tags = [tag_seq.split(',') for tag_seq in f.readlines()]
        
        max_seq_len = self.config_dic["max_seq_len"]
        PAD_WORD = self.config_dic['PAD_WORD']
        PAD_TAG = self.config_dic["PAD_TAG"]
        batch_size = self.config_dic['batch_size']
        retraining_epochs = self.config_dic["retraining_epochs"]

        batch_np_sequences = np.array(pad_word_sequence(corpus, max_seq_len, PAD_WORD))
        if batch_np_sequences.shape[0] % batch_size != 0:
            batch_pad = np.array([[PAD_WORD for _ in range (0, max_seq_len)] for _ in range(0,batch_size-(batch_np_sequences.shape[0] % batch_size))])
            batch_np_sequences = np.vstack((batch_np_sequences,batch_pad))

        tags_seq = [[self.tags2idx[t.strip("\n")] for t in s] for s in tags]
        batch_np_tags = pad_sequences(maxlen=max_seq_len, sequences=tags_seq, padding="post", value=self.tags2idx[PAD_TAG])
        if batch_np_tags.shape[0] % batch_size != 0:
            batch_pad = np.array([[self.tags2idx[PAD_TAG] for _ in range (0, max_seq_len)] for _ in range(0,batch_size-(batch_np_tags.shape[0] % batch_size))])
            batch_np_tags = np.vstack((batch_np_tags,batch_pad))
        
        assert (batch_np_sequences.shape == batch_np_tags.shape)

        batch_np_cat_tags = np.array([to_categorical(seq, num_classes=len(self.tags2idx)) for seq in batch_np_tags])

        print ("---------------start retraining----------------")
        t0 = time()
        with self.graph.as_default():
            with self.session.as_default():
                history = self.model.fit(batch_np_sequences, batch_np_cat_tags, validation_data=(batch_np_sequences, batch_np_cat_tags), batch_size=batch_size, 
                                epochs=retraining_epochs, validation_split=0.2, verbose=1, callbacks=[self.accuracy_training_callback])
        print(f'It took {time()- t0} seconds to retrain the model')
        
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save_weights(self.retrained_model_path)


        print ("---------------archiving learning sample----------------")
        data_cleaner = DataCleaner(os.path.realpath(f"./new_training_data/{self.appName}/archived_data/"))
        data_cleaner.archive_sample(self.retraining_text_seq_path, self.retraining_tags_seq_path)
        
        return json.dumps(history.history, cls=NumpyEncoder, sort_keys=True, indent=2)


    def train_on_archive (self):
        with open(self.retraining_archive_text_seq_path,"r", encoding="utf-8") as f:
            corpus = [sent.split() for sent in f.readlines()]
        with open(self.retraining_archive_tags_seq_path,"r") as f:
            tags = [tag_seq.split(',') for tag_seq in f.readlines()]
        
        max_seq_len = self.config_dic["max_seq_len"]
        PAD_WORD = self.config_dic['PAD_WORD']
        PAD_TAG = self.config_dic["PAD_TAG"]
        batch_size = self.config_dic['batch_size']
        retraining_epochs = self.config_dic["archive_retraining_epochs"]

        tagged_sequences = list(filter (lambda seq: any([True if tag.strip() != 'O' else False for tag in seq[1]]), list(zip(corpus,tags))))
        corpus = [seq[0] for seq in tagged_sequences]
        tags = [seq[1] for seq in tagged_sequences]

        batch_np_sequences = np.array(pad_word_sequence(corpus, max_seq_len, PAD_WORD))
        if batch_np_sequences.shape[0] % batch_size != 0:
            batch_pad = np.array([[PAD_WORD for _ in range (0, max_seq_len)] for _ in range(0,batch_size-(batch_np_sequences.shape[0] % batch_size))])
            batch_np_sequences = np.vstack((batch_np_sequences,batch_pad))

        tags_seq = [[self.tags2idx[t.strip("\n")] for t in s] for s in tags]
        batch_np_tags = pad_sequences(maxlen=max_seq_len, sequences=tags_seq, padding="post", value=self.tags2idx[PAD_TAG])
        if batch_np_tags.shape[0] % batch_size != 0:
            batch_pad = np.array([[self.tags2idx[PAD_TAG] for _ in range (0, max_seq_len)] for _ in range(0,batch_size-(batch_np_tags.shape[0] % batch_size))])
            batch_np_tags = np.vstack((batch_np_tags,batch_pad))
        
        assert (batch_np_sequences.shape == batch_np_tags.shape)

        batch_np_cat_tags = np.array([to_categorical(seq, num_classes=len(self.tags2idx)) for seq in batch_np_tags])

        print ("---------------start retraining----------------")
        t0 = time()
        with self.graph.as_default():
            with self.session.as_default():
                history = self.model.fit(batch_np_sequences, batch_np_cat_tags, validation_data=(batch_np_sequences, batch_np_cat_tags), batch_size=batch_size, 
                                epochs=retraining_epochs, validation_split=0.2, verbose=1, callbacks=[self.accuracy_training_callback])
        print(f'It took {time()- t0} seconds to retrain the model')
        
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save_weights(self.archive_retrained_model_path)
        
        return json.dumps(history.history, cls=NumpyEncoder, sort_keys=True, indent=2)
        

def test_extract_FB ():
    text="""I want to travel from Paris to Berlin from 2nd February 2020 to 5th March 2020.
I want to travel to Berlin from Barcelona from 2nd February 2020 to 5th March 2020.
I want to travel to London from New York from 2nd March 2020 to 22th June 2020.
I want to travel to New York from San Francisco from 2nd July 2020 to 22th August 2020.
I want to travel to New York from San Francisco from 2nd July 2020 to 22th August 2020.
I would like to fly from Los Angeles to Miami from 15th March 2020 to 20th June 2020.
I would like to fly from New York to Seoul from 5th July 2020 to 20th September 2020.
I would like to travel to Paris from Miami from 15th March 2020 to 20/10/2020."""
    my_extractor = NERExtractor(appName="FlightBooking")
    my_extractor.predict(text)

def test_retrain_FB ():
    text="""I would like to travel from Paris to Berlin from 1st January 2019 to 24th January 2020.
I would like to fly from Rome to Munich from 2nd March 2020 to 24th June 2020.
I want to book a flight to London from Paris from 3rd February 2020 to 15th Mai 2020.
I want to travel to Bucharest from Madrid from 06/01/2020 to 02/04/2020.
I am willing to fly from Bucharest to Barcelona from 03/03/2020 to 07/07/2020.
I wish to travel from Porto to Marseille from 4th April 2020 to 6th August 2020."""
    validation_results = """{"DocumentId":"filght_request_sample0_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":485},"Language":"","DocumentGroup":"FlightRequest","DocumentCategory":"FlightRequest","DocumentTypeId":"FlightRequest.FlightRequest.Request","DocumentTypeName":"Request","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"Request","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"FlightRequest.FlightRequest.Request.DepartureCity","FieldName":"DepartureCity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Paris","Reference":{"TextStartIndex":28,"TextLength":5,"Tokens":[{"TextStartIndex":28,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[74.18,189.9312,21.7046,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Rome","Reference":{"TextStartIndex":113,"TextLength":4,"Tokens":[{"TextStartIndex":113,"TextLength":4,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[96.62,175.22,26.0102,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Paris","Reference":{"TextStartIndex":206,"TextLength":5,"Tokens":[{"TextStartIndex":206,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[119.18,244.933,21.6163,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Madrid","Reference":{"TextStartIndex":288,"TextLength":6,"Tokens":[{"TextStartIndex":288,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[141.62,226.0651,32.7005,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Bucharest","Reference":{"TextStartIndex":351,"TextLength":9,"Tokens":[{"TextStartIndex":351,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[164.18,174.9168,44.6458,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Porto","Reference":{"TextStartIndex":427,"TextLength":5,"Tokens":[{"TextStartIndex":427,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[186.62,164.7269,24.7738,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"FlightRequest.FlightRequest.Request.ArrivalCity","FieldName":"ArrivalCity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Berlin","Reference":{"TextStartIndex":37,"TextLength":6,"Tokens":[{"TextStartIndex":37,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[74.18,226.032,26.1979,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Munich","Reference":{"TextStartIndex":121,"TextLength":6,"Tokens":[{"TextStartIndex":121,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[96.62,215.69,34.0032,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"London","Reference":{"TextStartIndex":194,"TextLength":6,"Tokens":[{"TextStartIndex":194,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[119.18,184.5474,33.6499,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Bucharest","Reference":{"TextStartIndex":273,"TextLength":9,"Tokens":[{"TextStartIndex":273,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[141.62,154.6694,44.6348,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Barcelona","Reference":{"TextStartIndex":364,"TextLength":9,"Tokens":[{"TextStartIndex":364,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[164.18,234.0691,44.632,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Marseille","Reference":{"TextStartIndex":436,"TextLength":9,"Tokens":[{"TextStartIndex":436,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[186.62,203.941,41.3779,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"FlightRequest.FlightRequest.Request.DepartureDate","FieldName":"DepartureDate","FieldType":"Date","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"1st January 2019","Reference":{"TextStartIndex":49,"TextLength":16,"Tokens":[{"TextStartIndex":49,"TextLength":16,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[73.28,279.05,10.7316,11.94],[74.18,292.25,59.0971,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"1"},{"FieldId":"Month","Value":"1"},{"FieldId":"Year","Value":"2019"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"2nd March 2020","Reference":{"TextStartIndex":133,"TextLength":14,"Tokens":[{"TextStartIndex":133,"TextLength":14,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[95.72,276.29,13.014,11.94],[96.62,291.77,53.7077,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"2"},{"FieldId":"Month","Value":"3"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"3rd February 2020","Reference":{"TextStartIndex":217,"TextLength":17,"Tokens":[{"TextStartIndex":217,"TextLength":17,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[118.28,293.2772,11.7468,11.94],[119.18,307.37,64.86,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"3"},{"FieldId":"Month","Value":"2"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"06/01/2020","Reference":{"TextStartIndex":300,"TextLength":10,"Tokens":[{"TextStartIndex":300,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[141.62,285.4897,53.0376,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"1"},{"FieldId":"Month","Value":"6"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"03/03/2020","Reference":{"TextStartIndex":379,"TextLength":10,"Tokens":[{"TextStartIndex":379,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[164.18,305.4178,53.0914,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"3"},{"FieldId":"Month","Value":"3"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"4th April 2020","Reference":{"TextStartIndex":451,"TextLength":14,"Tokens":[{"TextStartIndex":451,"TextLength":14,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[185.72,272.21,11.574,11.94],[186.62,286.37,45.7719,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"4"},{"FieldId":"Month","Value":"4"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"FlightRequest.FlightRequest.Request.ArrivalDate","FieldName":"ArrivalDate","FieldType":"Date","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"24th January 2020.","Reference":{"TextStartIndex":69,"TextLength":18,"Tokens":[{"TextStartIndex":69,"TextLength":18,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[73.28,365.7543,17.1697,11.94],[74.18,385.51,62.0338,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"24"},{"FieldId":"Month","Value":"1"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"24th June 2020.","Reference":{"TextStartIndex":151,"TextLength":15,"Tokens":[{"TextStartIndex":151,"TextLength":15,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[95.72,359.938,17.226,11.94],[96.62,379.75,48.1133,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"24"},{"FieldId":"Month","Value":"6"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"15th Mai 2020.","Reference":{"TextStartIndex":238,"TextLength":14,"Tokens":[{"TextStartIndex":238,"TextLength":14,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[118.28,386.6372,17.1668,11.94],[119.18,406.39,17.2997,11.04],[119.18,425.9418,25.2303,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":""},{"FieldId":"Month","Value":""},{"FieldId":"Year","Value":""}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"02/04/2020.","Reference":{"TextStartIndex":314,"TextLength":11,"Tokens":[{"TextStartIndex":314,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[141.62,352.9455,55.8845,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"4"},{"FieldId":"Month","Value":"2"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"07/07/2020.","Reference":{"TextStartIndex":393,"TextLength":11,"Tokens":[{"TextStartIndex":393,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[164.18,372.9274,55.8845,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":"7"},{"FieldId":"Month","Value":"7"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"August 2020.","Reference":{"TextStartIndex":473,"TextLength":12,"Tokens":[{"TextStartIndex":473,"TextLength":12,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[186.62,360.79,58.5672,11.04]]}]},"DerivedFields":[{"FieldId":"Day","Value":""},{"FieldId":"Month","Value":"8"},{"FieldId":"Year","Value":"2020"}],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1}]}}"""
    my_extractor = NERExtractor(appName="FlightBooking")
    my_extractor.retrain(text,validation_results)
    
def test_retrain_EAP ():
    text="""Dear Sir and/or Madam: I am writing on behalf of Nicholas Galletti, who ﬁled a complaint with our ofﬁce regarding the enclosed June 27, 2019 billing statement from COmenity Bank for his Alphaeon credit card.
The billing statement indicates he owes a balance of $702.05, to be due on July 20, 2019.
The credit card payments are to pay off a medical bill.
MI. Galletti is protesting this charge, as his June 10, 2019 statement indicates that he owed $1,200, to be due on June 20, 2019.
Mr. Galletti states that he made the payment of $1,200 on June 14, 2019, and believes his account should have a zero balance. . . . . ~ , Mr.
Galletti states that he paid each bill monthly in advance of the due date.
He States that he cantacted Comenity Bank representatives, who notiﬁed him that the new balance was for charged interest because his bill was late.
He states that the representatives told him the ﬁnal bill was due on June 6, 2019.
Mr. Galletti points out that the enclosed June 10, 2019 statement indicates that the $1,200 balance had a June 20, 2019 due date, which is contradictory information than the information provided by the representatives, and that his payment was, in fact, made before the due date.
1 ask that Comenity Bank review Mr.
Galletti's account, remove the charges since he paid the bill off in advance of the due date indicated on the statement, and provide a zero balance statement.
If this request cannot be granted, please provide a detailed written explanation, including the portion of Mr.
Galletti's credit card agreement indicating that the ﬁnal payment was due on June 6, 2019, and information indicating how the charges were calculated.
1 ask for a written reply to my attention within 30 calendar days via fax at (518) 650-9365 or by mail to Matthew Corvin, Ofﬁce of the Attorney General, Health Care Bureau.
The Capitol, Albany, NY 122240341.
If you have any questions, feel free to contact me at (518) 776-2473.
Thank you for your attention to this matter.
Very truly yours,"""
    validation_results = """{"DocumentId":"0707_001_ocr_text_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":2009},"Language":"","DocumentGroup":"EAP","DocumentCategory":"ComenityBankLetter","DocumentTypeId":"EAP.ComenityBankLetter.ConsumerAffair","DocumentTypeName":"ConsumerAffair","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"ConsumerAffair","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"EAP.ComenityBankLetter.ConsumerAffair.deadline","FieldName":"deadline","FieldType":"Text","IsMissing":false,"DataSource":"Manual","Values":[{"Components":[],"Value":"30 calendar days","Reference":{"TextStartIndex":1718,"TextLength":16,"Tokens":[{"TextStartIndex":1718,"TextLength":16,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[530.62,283.727,11.2277,11.04],[530.62,297.3835,38.5517,11.04],[530.62,338.4302,20.3909,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1}]}}"""
    my_extractor = NERExtractor(appName="EAP")
    my_extractor.retrain(text,validation_results)




def test_extract_GMB():
    my_extractor = NERExtractor(appName="GMB")
    # for i in range (20,25):
    #     my_classifier.predict(test_texts[i])#, test_tags_path.split()
    sample = """HARTLEPOOL, England - It used to be simple in Hartlepool.

Come election time, a majority of voters in this coastal working class town in the northeast of England would inevitably back the left-leaning Labour Party.
Proud of its industrial history as a center for shipbuilding - the H.M.S. Trincomalee, Europe's oldest floating warship, sits in the town's windswept marina - this is still regarded a Labour heartland as the United Kingdom prepares for an election Dec.
12.

But old ties are being tested to their limits with a possibly devastating impact on Labour, which is fighting to defeat the ruling Conservative Party.
Voters in Hartlepool feel they have missed out on the economic growth of the last few decades, which has transformed London and the southeast of England.
This is the eastern edge of what pollsters have called the "red wall" of Brexit-supporting seats Labour holds with a majority of less than 8,000 votes.

And there's another problem facing many traditional Labour voters here: the party's leader, Jeremy Corbyn.

"I don't think he's a good leader," said Christine Scott, 57, as she prepared some herring for sale in Hodgson's fishmongers, part of a family-owned business that's been in Hartlepool for more than a century. "I'm not 100 percent sure about him.\""""
    return my_extractor.predict(sample)

def test_retrain_GMB():

    text = """(Bloomberg) -- China took action on two fronts to gain control of the spiraling coronavirus outbreak gripping the country: reporting a dramatic increase in cases and ousting top officials who failed to check the disease's expansion.

The moves came within hours of each other Thursday.
First, health authorities revealed that cases in Hubei province, where the disease first emerged, had surged by 45% to almost 50,000 after they included a new group of patients.
That raised the global tally to almost 60,000, dashing hopes that the epidemic might be easing.

Then, authorities announced the replacement of the two most senior Communist Party officials in Hubei and its hard-hit capital, Wuhan.
Shanghai Mayor Ying Yong -- a former top judge who once served under President Xi Jinping -- was named to replace embattled provincial boss, Jiang Chaoliang.

The shakeup represented the biggest political fallout yet from an outbreak that has in a little over two months shaken confidence in China's leaders and caused countries from the U.S. to Japan to restrict or block visits by its citizens.
The decision on the heels of the expanded case tally was reminiscent of a boardroom reset, in which bad numbers are disclosed first to give the new team a clean slate.

"First of all, they are trying to clean up the backlog of untested people who haven't been confirmed," said Ether Yin, partner at Beijing-based consulting firm Trivium China. "It's politically important for them to get the number right before the new party boss comes in, so that all the new confirmed cases happened under Jiang Chaoliang's watch.
The new party secretary needs a new starting line."

The surprise revision came amid mounting speculation that China was undercounting cases of the new coronavirus strain, as countries around the world struggle to contain a disease that appears to spread when patients show only mild symptoms.
Daily declines in new cases in Hubei earlier this week helped push U.S. equity markets to record highs on Wednesday."""
    validation_results="""{"DocumentId":"china_virus_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":2020},"Language":"","DocumentGroup":"NewsArticle","DocumentCategory":"Politics","DocumentTypeId":"NewsArticle.Politics.article","DocumentTypeName":"article","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"article","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"NewsArticle.Politics.article.Person","FieldName":"Person","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Ying Yong","Reference":{"TextStartIndex":711,"TextLength":9,"Tokens":[{"TextStartIndex":711,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[207.068,427.142,24.108,11.676],[207.068,454.142,27.084,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"President Xi Jinping","Reference":{"TextStartIndex":765,"TextLength":20,"Tokens":[{"TextStartIndex":765,"TextLength":20,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[220.658,288.588,50.774,11.676],[220.658,342.29,12.036,11.676],[220.658,357.17,40.416,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Jiang","Reference":{"TextStartIndex":837,"TextLength":5,"Tokens":[{"TextStartIndex":837,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[234.338,214.176,28.908,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Ether Yin","Reference":{"TextStartIndex":1370,"TextLength":9,"Tokens":[{"TextStartIndex":1370,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[353.758,162.12,29.616,11.676],[353.758,194.724,21.24,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Jiang","Reference":{"TextStartIndex":1585,"TextLength":5,"Tokens":[{"TextStartIndex":1585,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[380.998,365.288,29.016,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.TimeOrDate","FieldName":"TimeOrDate","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Thursday","Reference":{"TextStartIndex":276,"TextLength":8,"Tokens":[{"TextStartIndex":276,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[126.788,304.556,53.52,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Wednesday","Reference":{"TextStartIndex":2010,"TextLength":9,"Tokens":[{"TextStartIndex":2010,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[474.958,88.476,64.428,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.Organization","FieldName":"Organization","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Bloomberg","Reference":{"TextStartIndex":1,"TextLength":9,"Tokens":[{"TextStartIndex":0,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[73.988,72.024,67.368,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Communist Party","Reference":{"TextStartIndex":628,"TextLength":15,"Tokens":[{"TextStartIndex":628,"TextLength":15,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[193.388,436.644,62.1,11.676],[193.388,501.684,28.32,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Trivium","Reference":{"TextStartIndex":1422,"TextLength":7,"Tokens":[{"TextStartIndex":1422,"TextLength":7,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[353.758,434.174,42.816,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeographicEntity","FieldName":"GeographicEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"China","Reference":{"TextStartIndex":15,"TextLength":5,"Tokens":[{"TextStartIndex":15,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[73.988,154.1,31.26,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hubei","Reference":{"TextStartIndex":335,"TextLength":5,"Tokens":[{"TextStartIndex":335,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[140.468,140.172,32.748,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hubei","Reference":{"TextStartIndex":657,"TextLength":5,"Tokens":[{"TextStartIndex":657,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[207.068,129.732,32.748,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Shanghai","Reference":{"TextStartIndex":696,"TextLength":8,"Tokens":[{"TextStartIndex":696,"TextLength":8,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[207.068,337.322,49.44,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"U.S.","Reference":{"TextStartIndex":1034,"TextLength":4,"Tokens":[{"TextStartIndex":1034,"TextLength":4,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[287.258,91.824,22.308,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Japan","Reference":{"TextStartIndex":1042,"TextLength":5,"Tokens":[{"TextStartIndex":1042,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[287.258,130.548,32.172,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"China","Reference":{"TextStartIndex":1721,"TextLength":5,"Tokens":[{"TextStartIndex":1721,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[420.358,390.0,31.248,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hubei","Reference":{"TextStartIndex":1935,"TextLength":5,"Tokens":[{"TextStartIndex":1935,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[461.278,154.092,32.748,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"U.S.","Reference":{"TextStartIndex":1971,"TextLength":4,"Tokens":[{"TextStartIndex":1971,"TextLength":4,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[461.278,346.92,22.308,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Wuhan","Reference":{"TextStartIndex":689,"TextLength":5,"Tokens":[{"TextStartIndex":689,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[207.068,292.562,41.868,11.676]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeopoliticalEntity","FieldName":"GeopoliticalEntity","FieldType":"Text","IsMissing":true,"DataSource":"Automatic","Values":[],"DataVersion":0}]}}"""
    my_extractor = NERExtractor("GMB")
    my_extractor.retrain(text,validation_results)

def test_archive_train_FB():
    my_extractor = NERExtractor(appName="FlightBooking")
    my_extractor.train_on_archive()

def test_archive_train_EAP():
    my_extractor = NERExtractor(appName="EAP")
    my_extractor.train_on_archive()


if __name__=='__main__':
    # sample = test_predict()
    # test_retrain_GMB()
    # test_retrain_FB()
    test_extract_FB()
    # test_archive_train_FB()
    # test_retrain_EAP()
    # test_archive_train_EAP()