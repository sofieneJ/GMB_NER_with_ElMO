# This model predicts NERs amongst {org, per, tim, geo(Geographical Entity), gpe(Geopolitical Entity)}
# This work is based on Tobias Sterbak' work https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/
# @author : sofiene.jenzri@uipath.com
# @version: 1.0 05/12/2019

# cf https://www.kaggle.com/navya098/bi-lstm-for-ner
import numpy as np # linear algebra
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
import json

from nltk.tokenize import sent_tokenize


def build_model(pretrained_ElMO_module, max_seq_len, num_tags, batch_size):    
 
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
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
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
    def __init__(self):
        
        pretrained_ElMO_path =  os.path.realpath("./models/ElMO/hub_elmo/")
        trained_model_path =  os.path.realpath("./models/ElMO/ElMo_BiLSTM_keras.h5")
        tags_dic_path = os.path.realpath('./models/ElMO/tags_dic.json')
        model_config_path = os.path.realpath('./models/ElMO/model_config.json')        


        with open(tags_dic_path, 'r') as f:
            self.tags2idx=json.load(f)
        with open(model_config_path, 'r') as f:
            self.config_dic=json.load(f)

        self.session = tf.Session()
        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            with self.session.as_default():
                pretrained_elmo_model = hub.Module(spec=pretrained_ElMO_path, trainable=True)
                self.model = build_model(pretrained_ElMO_module=pretrained_elmo_model, max_seq_len=self.config_dic["max_seq_len"], 
                                        num_tags=len(self.tags2idx), batch_size=self.config_dic["batch_size"])
                self.model.load_weights(trained_model_path)
                                        
        self.idx2tag = {idx:tag for tag,idx in self.tags2idx.items()} 


    def predict(self, input):
        #Load model params
        max_seq_len = self.config_dic["max_seq_len"]
        PAD_WORD = self.config_dic['PAD_WORD']
        batch_size = self.config_dic['batch_size']
        
        sents = sent_tokenize(input)
        sents = [sent for sent in sents if len(sent.split())<max_seq_len]
        nb_sentences = len(sents)


        np_sequences = np.empty((0,max_seq_len))
        np_predictions = np.empty((0,max_seq_len))
        np_probas = np.empty((0,max_seq_len))

        for i in range (0, (nb_sentences//batch_size)+1):
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

            with self.graph.as_default():
                with self.session.as_default():
                    p = self.model.predict(batch_np_sequences, batch_size=batch_size)
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
                    start_index = len(' '.join(np_sequences[k][:i]))+ input.index(sent, sum([len(sentence) for h,sentence in enumerate(sents) if h<k]))
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
                        "entity_text":entity_text,
                        "start_index":start_index,
                        "entity_length":entity_len,
                        "proba":proba
                    }
                    predicted_entities.append(entity)

        print (json.dumps(predicted_entities, sort_keys=True, indent=2) )
        return json.dumps(predicted_entities, sort_keys=True, indent=2)


def test_main():

    # test_corpus_path = './temp_data/test_copus.txt'
    # with open(test_corpus_path, "r", encoding = "utf8") as f:
    #     test_texts = f.readlines()
    #     print (f'test corpus contains {len(test_texts)} sequences')
    # # with open(test_tags_path, "r", encoding = "utf8") as f:
    # #     test_tags = f.readlines()
    # test_texts = sorted(test_texts, key=lambda text:len(text), reverse=True)
    my_extractor = NERExtractor()
    # for i in range (20,25):
    #     my_classifier.predict(test_texts[i])#, test_tags_path.split()
    sample = """                Some Title
An Australian academic freed by the Taliban in a prisoner swap has spoken of his "long and tortuous ordeal" as a hostage in Afghanistan.

Malta police arrested one of the country's most prominent businessmen on Wednesday in connection with an investigation into the murder of journalist Daphne Caruana Galizia."""
    sample = """Malta police arrested one of the country's most prominent businessmen on Wednesday in connection with an investigation into the murder of journalist Daphne Caruana Galizia.
Yorgen Fenech was detained after police intercepted his yacht off the Mediterranean island, sources with knowledge of the matter said.
Fenech is a director and co-owner of a business group that won a large energy concession in 2013 from the Maltese state to build a gas power station on the island.
His luxury yacht Gio left the Portomaso yacht marina, five miles (eight km) north of Valletta, shortly before dawn. Police swiftly boarded the vessel and forced it to return to port.
A yacht, which is believed to have been intercepted by Maltese police to arrest prominent businessman Yorgen FenechHandout via REUTERS
His arrest came the day after the government said it would offer a pardon to a suspected middleman in the 2017 murder of Caruana Galizia if he named the mastermind behind the killing.
The middleman is believed to have linked the person suspected of commissioning her murder, to the men accused of carrying out the killing, as well as those who helped procure the explosive device used in the operation.
The Prime Minister's Office said that the man "showed willingness, after some time of interrogation by the police, to collaborate" but that he "asked to be granted a presidential pardon first" for all the cases he might have been involved in.
The island's leader added that after some negotiations with the man's lawyers, he drafted and signed a letter giving the suspect assurances that if he "gave all the information and evidence that he had, and if all this could be corroborated in court, I would recommend that this person be given a presidential pardon."
Yorgen Fenech of the Tumas Group talks with VIP guests during the opening of the Oracle Casino in St Paul's Bay, Malta, June 4, 2014.REUTERS
Caruana Galizia, a well-known investigative journalist who wrote an anti-corruption blog, was killed by a car bomb near the Maltese capital Valletta in October 2017 - a murder that shocked Europe and raised questions about the rule of law on the Mediterranean island.
The arrest was carried out on Thursday morning during a Europol-backed operation into money laundering, the statement from the Prime Minister's Office confirmed.
Malta police arrested one of the country's most prominent businessmen on Wednesday in connection with an investigation into the murder of journalist Daphne Caruana Galizia.
Yorgen Fenech was detained after police intercepted his yacht off the Mediterranean island, sources with knowledge of the matter said.
Fenech is a director and co-owner of a business group that won a large energy concession in 2013 from the Maltese state to build a gas power station on the island.
His luxury yacht Gio left the Portomaso yacht marina, five miles (eight km) north of Valletta, shortly before dawn. Police swiftly boarded the vessel and forced it to return to port.
A yacht, which is believed to have been intercepted by Maltese police to arrest prominent businessman Yorgen FenechHandout via REUTERS
His arrest came the day after the government said it would offer a pardon to a suspected middleman in the 2017 murder of Caruana Galizia if he named the mastermind behind the killing.
The middleman is believed to have linked the person suspected of commissioning her murder, to the men accused of carrying out the killing, as well as those who helped procure the explosive device used in the operation.
The Prime Minister's Office said that the man "showed willingness, after some time of interrogation by the police, to collaborate" but that he "asked to be granted a presidential pardon first" for all the cases he might have been involved in.
The island's leader added that after some negotiations with the man's lawyers, he drafted and signed a letter giving the suspect assurances that if he "gave all the information and evidence that he had, and if all this could be corroborated in court, I would recommend that this person be given a presidential pardon."
Yorgen Fenech of the Tumas Group talks with VIP guests during the opening of the Oracle Casino in St Paul's Bay, Malta, June 4, 2014.REUTERS
Caruana Galizia, a well-known investigative journalist who wrote an anti-corruption blog, was killed by a car bomb near the Maltese capital Valletta in October 2017 - a murder that shocked Europe and raised questions about the rule of law on the Mediterranean island.
The arrest was carried out on Thursday morning during a Europol-backed operation into money laundering, the statement from the Prime Minister's Office confirmed.
Malta police arrested one of the country's most prominent businessmen on Wednesday in connection with an investigation into the murder of journalist Daphne Caruana Galizia.
Yorgen Fenech was detained after police intercepted his yacht off the Mediterranean island, sources with knowledge of the matter said.
Fenech is a director and co-owner of a business group that won a large energy concession in 2013 from the Maltese state to build a gas power station on the island.
His luxury yacht Gio left the Portomaso yacht marina, five miles (eight km) north of Valletta, shortly before dawn. Police swiftly boarded the vessel and forced it to return to port.
A yacht, which is believed to have been intercepted by Maltese police to arrest prominent businessman Yorgen FenechHandout via REUTERS
His arrest came the day after the government said it would offer a pardon to a suspected middleman in the 2017 murder of Caruana Galizia if he named the mastermind behind the killing.
The middleman is believed to have linked the person suspected of commissioning her murder, to the men accused of carrying out the killing, as well as those who helped procure the explosive device used in the operation.
The Prime Minister's Office said that the man "showed willingness, after some time of interrogation by the police, to collaborate" but that he "asked to be granted a presidential pardon first" for all the cases he might have been involved in.
The island's leader added that after some negotiations with the man's lawyers, he drafted and signed a letter giving the suspect assurances that if he "gave all the information and evidence that he had, and if all this could be corroborated in court, I would recommend that this person be given a presidential pardon."
Yorgen Fenech of the Tumas Group talks with VIP guests during the opening of the Oracle Casino in St Paul's Bay, Malta, June 4, 2014.REUTERS
Caruana Galizia, a well-known investigative journalist who wrote an anti-corruption blog, was killed by a car bomb near the Maltese capital Valletta in October 2017 - a murder that shocked Europe and raised questions about the rule of law on the Mediterranean island.
The arrest was carried out on Thursday morning during a Europol-backed operation into money laundering, the statement from the Prime Minister's Office confirmed."""

    my_extractor.predict(sample)
    return sample



if __name__=='__main__':
    sample = test_main()
    print(sample[1062:1073])

