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
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])
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
        taxonomy_map_path = os.path.realpath('./models/ElMO/taxonomy_mapping.json')
        self.retraining_text_seq_path = os.path.realpath('./new_training_data/hot_folder/text_sequences.txt')
        self.retraining_tags_seq_path = os.path.realpath('./new_training_data/hot_folder/tags_sequences.txt')

        with open(tags_dic_path, 'r') as f:
            self.tags2idx=json.load(f)
        with open(model_config_path, 'r') as f:
            self.config_dic=json.load(f)
        with open(taxonomy_map_path, 'r') as f:
            self.taxo_map = json.load(f)

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
        
        sents = sent_tokenize(text)
        sents = [sent for sent in sents if len(sent.split())<max_seq_len]
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
             
            with open(self.retraining_text_seq_path,"a") as f:
                f.write(sent)
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
        with open(self.retraining_text_seq_path,"r") as f:
            corpus = [sent.split() for sent in f.readlines()]
        with open(self.retraining_tags_seq_path,"r") as f:
            tags = [tag_seq.split(',') for tag_seq in f.readlines()]
        
        max_seq_len = self.config_dic["max_seq_len"]
        PAD_WORD = self.config_dic['PAD_WORD']
        PAD_TAG = self.config_dic["PAD_TAG"]
        batch_size = self.config_dic['batch_size']

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
                                epochs=2, validation_split=0.2, verbose=1)
        print(f'It took {time()- t0} seconds to retrain the model')
        
        re_trained_model_path =  os.path.realpath("./models/ElMO/retrained_ElMo_BiLSTM_keras.h5")
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save_weights(re_trained_model_path)


        print ("---------------archiving learning sample----------------")
        data_cleaner = DataCleaner(os.path.realpath('./new_training_data/archived_data/'))
        data_cleaner.archive_sample(self.retraining_text_seq_path, self.retraining_tags_seq_path)
        
        return json.dumps(history.history, cls=NumpyEncoder, sort_keys=True, indent=2)
        
                    
def test_retrain ():
    text="""Hundreds of thousands of demonstrators poured into the Hong Kong streets on Sunday in a mass show of support -- marking sixth months of pro-democracy protests and highlighting the resilience of a people who continue to fight for their freedom and autonomy against the Chinese government.
Chanting "Fight for freedom" and "Stand with Hong Kong," the protesters formed a mile-long human snake that winded through blocks from the Causeway Bay shopping district to the Central business zone.
The crowds were reportedly so large that the group was forced to pause at times.
Organizers said 800,000 people participated, although local police didn't have an exact figure.

One of the protesters, however, was nearly hidden from view.

A young woman was seen crawling on her hands and knees on rough streets -- a metaphor for the arduous path and continuous fighting that pro-democracy protesters have faced in order to ensure their eventual freedom.

"This is just the beginning.
We have a long way to run," Louisa Yiu, an engineer and protester, told the Guardian.
The crawling protester also dragged bricks and empty soda cans on a string behind her -- another metaphor for the weight they've been carrying -- which excited fellow protesters who were heard yelling "Go for it!"

"Her performance art is about the difficulty, or the repetitiveness, of demonstrations," said one of her friends, who walked alongside and identified herself by her surname, Chan. "This is really a long-term struggle."

Marchers were captured holding up five fingers, a symbol for the protest movement's five demands.
They include democratic elections and an investigation into the actions of police throughout the last six months of protests, according to the Guardian.
The belief was that the protester movement would cease over time as it enters the seventh month.
It hasn't."""
    validation_results = """{"DocumentId":"hongkong_v.pdf","ResultsVersion":1,"ResultsDocument":{"Bounds":{"StartPage":0,"PageCount":1,"TextStartIndex":0,"TextLength":1852},"Language":"","DocumentGroup":"NewsArticle","DocumentCategory":"Politics","DocumentTypeId":"NewsArticle.Politics.article","DocumentTypeName":"article","DocumentTypeDataVersion":0,"DataVersion":1,"DocumentTypeSource":"Automatic","DocumentTypeField":{"Components":[],"Value":"article","Reference":{"TextStartIndex":0,"TextLength":0,"Tokens":[]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},"Fields":[{"FieldId":"NewsArticle.Politics.article.Person","FieldName":"Person","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Louisa Yiu,","Reference":{"TextStartIndex":1001,"TextLength":11,"Tokens":[{"TextStartIndex":1001,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[397.03,333.7382,32.4753,11.04],[397.03,368.819,17.2113,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Chan.","Reference":{"TextStartIndex":1448,"TextLength":5,"Tokens":[{"TextStartIndex":1448,"TextLength":5,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[544.06,419.3624,27.7988,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.TimeOrDate","FieldName":"TimeOrDate","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Sunday","Reference":{"TextStartIndex":76,"TextLength":6,"Tokens":[{"TextStartIndex":76,"TextLength":6,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[81.98,457.3367,35.8138,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.Organization","FieldName":"Organization","FieldType":"Text","IsMissing":false,"DataSource":"Manual","Values":[{"Components":[],"Value":"Guardian.","Reference":{"TextStartIndex":1049,"TextLength":9,"Tokens":[{"TextStartIndex":1049,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[418.03,90.024,46.3146,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Guardian.","Reference":{"TextStartIndex":1735,"TextLength":9,"Tokens":[{"TextStartIndex":1735,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[649.08,359.5166,46.2798,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeographicEntity","FieldName":"GeographicEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Hong Kong","Reference":{"TextStartIndex":55,"TextLength":9,"Tokens":[{"TextStartIndex":55,"TextLength":9,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[81.98,349.99,26.0323,11.04],[81.98,378.67,25.3147,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Hong Kong,'","Reference":{"TextStartIndex":333,"TextLength":11,"Tokens":[{"TextStartIndex":333,"TextLength":11,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[165.98,288.3749,26.0433,11.04],[165.98,317.0458,30.6139,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0},{"Components":[],"Value":"Causeway Bay","Reference":{"TextStartIndex":427,"TextLength":12,"Tokens":[{"TextStartIndex":427,"TextLength":12,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[186.98,316.6704,49.7462,11.04],[186.98,369.0773,17.8517,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1},{"FieldId":"NewsArticle.Politics.article.GeopoliticalEntity","FieldName":"GeopoliticalEntity","FieldType":"Text","IsMissing":false,"DataSource":"ManuallyChanged","Values":[{"Components":[],"Value":"Chinese","Reference":{"TextStartIndex":268,"TextLength":7,"Tokens":[{"TextStartIndex":268,"TextLength":7,"Page":0,"PageWidth":612.0,"PageHeight":792.0,"Boxes":[[144.98,90.024,38.5517,11.04]]}]},"DerivedFields":[],"Confidence":1.0,"OperatorConfirmed":true,"OcrConfidence":1.0}],"DataVersion":1}]}}"""
    my_extractor = NERExtractor()
    my_extractor.retrain(text,validation_results)




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
    

