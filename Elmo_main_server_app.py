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
        taxonomy_map_path = './models/taxonomy_mapping.json'
        with open(taxonomy_map_path, 'r') as taxo_file:
            taxo_map = json.load(taxo_file)

        json_extraction = json.loads(validation_results)
        tags_list =[]
        for entity_group in json_extraction["ResultsDocument"]["Fields"]:
            for entity in entity_group["values"]:
                entity_text = entity["Value"]
                start_index = entity["Reference"]["TextStartIndex"]
                entity_text_length =  entity["Reference"]["TextLength"]
                tag = taxo_map [entity_group["FieldId"]]
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
        
                    
def test_retrain ():
    text="""Indian police are being declared heroes Friday after they shot and killed four men suspected of raping and killing a young woman in southern India just a week earlier.

Last Thursday, the remains of a 27-year-old veterinarian were discovered by a passerby at an underpass in the town of Shadnager, near Hyderabad, after she left her scooter at a toll booth for a medical visit the night before.

The four men -- between the ages of 20 and 26 -- allegedly deflated her scooter and took her to a truck yard to repair it. Soon after, they assaulted and suffocated the woman before burning her body, according to Sky News."""
    validation_results = """{"DocumentId":"india_news_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":618},"Language":"","DocumentGroup":"NewsArticle","DocumentCategory":"Politics","DocumentTypeId":"NewsArticle.Politics.article","DocumentTypeName":"article","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"article","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"NewsArticle.Politics.article.Person","FieldName":"Person","FieldType":"Text","IsMissing":true,"DataSource":"Automatic","Values":[],"DataVersion":0},{"FieldId":"NewsArticle.Politics.article.TimeOrDate","FieldName":"TimeOrDate","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Friday","Reference":{"TextStartIndex":40,"TextLength":6,"Tokens":[{"TextStartIndex":40,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[74.18,250.5975,27.313,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Thursday,","Reference":{"TextStartIndex":174,"TextLength":9,"Tokens":[{"TextStartIndex":174,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[133.7,92.4149,43.895,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"night","Reference":{"TextStartIndex":381,"TextLength":5,"Tokens":[{"TextStartIndex":381,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[162.62,89.5114,22.93,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.Organization","FieldName":"Organization","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Sky News.","Reference":{"TextStartIndex":609,"TextLength":9,"Tokens":[{"TextStartIndex":609,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[236.57,130.0613,14.9812,11.04],[236.57,147.5707,27.5007,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeographicEntity","FieldName":"GeographicEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"southern India","Reference":{"TextStartIndex":132,"TextLength":14,"Tokens":[{"TextStartIndex":132,"TextLength":14,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[88.7,204.3274,40.6161,11.04],[88.7,247.3944,22.1352,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Shadnager,","Reference":{"TextStartIndex":287,"TextLength":10,"Tokens":[{"TextStartIndex":287,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[148.1,137.6126,50.205,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hyderabad,","Reference":{"TextStartIndex":303,"TextLength":10,"Tokens":[{"TextStartIndex":303,"TextLength":10,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[148.1,212.9998,51.866,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeopoliticalEntity","FieldName":"GeopoliticalEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Indian","Reference":{"TextStartIndex":0,"TextLength":6,"Tokens":[{"TextStartIndex":0,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[74.18,72.024,27.9091,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1}]}}"""

    taxonomy_map_path = './models/taxonomy_mapping.json'
    with open(taxonomy_map_path, 'r') as taxo_file:
        taxo_map = json.load(taxo_file)

    json_extraction = json.loads(validation_results)
    tags_list =[]
    for entity_group in json_extraction["ResultsDocument"]["Fields"]:
        for entity in entity_group["Values"]:
            entity_text = entity["Value"]
            start_index = entity["Reference"]["TextStartIndex"]
            entity_text_length =  entity["Reference"]["TextLength"]
            tag = taxo_map [entity_group["FieldId"]]
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
    
    max_seq_len = 128
    batch_size = 32    
    
    sents = sent_tokenize(text)
    sents = [sent for sent in sents if len(sent.split())<max_seq_len]
    nb_sentences = len(sents)

    retraining_text_seq = './new_training_data/hot_folder/text_sequences.txt'
    retraining_tags_seq = './new_training_data/hot_folder/tags_sequences.txt'

    nb_hits = 0
    for i,sent in enumerate(sents):
        tag_sequence= []
        for j,token in enumerate(sent.split()):
            index_in_text = text.index(sent, sum([len(s) for k,s in enumerate(sents) if k<i])) + sent.index(token, sum([(len(t)+1) for k,t in enumerate(sent.split()) if k<j ]))
            hits = list(filter(lambda tag: tag["start_index"]==index_in_text, tags_list))
            if len(hits) != 0:
                assert(len(hits)==1)
                assert (hits[0]["token"]==token)
                tag_sequence.append(hits[0]["tag"])
                nb_hits+=1
            else:
                tag_sequence.append("O")
        with open(retraining_text_seq,"a") as f:
            f.write(sent)
            f.write("\n")
        with open(retraining_tags_seq,"a") as f:
            f.write(','.join(tag_sequence))
            f.write("\n")
    assert (len(tags_list)==nb_hits)



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
    sample = """HARTLEPOOL, England - It used to be simple in Hartlepool.

Come election time, a majority of voters in this coastal working class town in the northeast of England would inevitably back the left-leaning Labour Party.
Proud of its industrial history as a center for shipbuilding - the H.M.S. Trincomalee, Europe's oldest floating warship, sits in the town's windswept marina - this is still regarded a Labour heartland as the United Kingdom prepares for an election Dec.
12.

But old ties are being tested to their limits with a possibly devastating impact on Labour, which is fighting to defeat the ruling Conservative Party.
Voters in Hartlepool feel they have missed out on the economic growth of the last few decades, which has transformed London and the southeast of England.
This is the eastern edge of what pollsters have called the "red wall" of Brexit-supporting seats Labour holds with a majority of less than 8,000 votes.

And there's another problem facing many traditional Labour voters here: the party's leader, Jeremy Corbyn.

"I don't think he's a good leader," said Christine Scott, 57, as she prepared some herring for sale in Hodgson's fishmongers, part of a family-owned business that's been in Hartlepool for more than a century. "I'm not 100 percent sure about him.\""""

    my_extractor.predict(sample)
    return sample



if __name__=='__main__':
    # sample = test_main()
    test_retrain()


    # taxonomy_dic = {
    #     "NewsArticle.Politics.article.GeopoliticalEntity":"-gpe",
    #     "NewsArticle.Politics.article.GeographicEntity":"-geo",
    #     "NewsArticle.Politics.article.Organization" :"-org",
    #     "NewsArticle.Politics.article.TimeOrDate" :"-tim",
    #     "NewsArticle.Politics.article.Person":"-per"
    # }
    
    # taxonomy_map_path = './models/taxonomy_mapping.json'
    # with open(taxonomy_map_path, 'w') as taxo_file:
    #     taxo_file.write(json.dumps(taxonomy_dic, sort_keys=True, indent=2))
    

