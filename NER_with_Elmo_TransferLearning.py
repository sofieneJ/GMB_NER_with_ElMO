import tensorflow_hub as hub
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Lambda
from keras.layers.merge import add


from keras import backend as K
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import json

trained_model_path = './models/keras_ElMo_biLSTM_GMB_ner.h5'
trained_tfsaver_path = './models/tf_session_model/tf_elmo_biLSTM_GMB_NER'
test_corpus_path = './temp_data/test_copus.txt'
test_tags_path = './temp_data/test_tags.txt'
tags_dic_path = './models/tags_dic.json'
model_config_path = './models/model_config.json'
sub_sample_ratio=0.01

class SentenceGetter(object):
    
    def __init__(self, dataset, sub_sample_ratio):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        ########################  SUBSAMPLE dataset #################################
        index = np.arange(0,self.grouped.shape[0])
        np.random.seed(101) 
        np.random.shuffle(index)
        index = index[0: int(sub_sample_ratio*index.shape[0])]
        self.grouped = self.grouped.iloc[index].reset_index(drop=True)

        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def process_data(bTrainMode, sub_sample_ratio):
    raw_data_path = '../datasets/GMB_ner/'
    dframe = pd.read_csv(raw_data_path+'ner.csv', encoding = "ISO-8859-1", error_bad_lines=False)

    dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word',"pos"],axis=1)
    
    dataset=dataset.drop(['shape'],axis=1)
    # print(dataset.head())


    getter = SentenceGetter(dataset, sub_sample_ratio)
    my_sentences = getter.sentences
    print (f'corpus contains {len(my_sentences)} sentences')

    max_seq_len = max([len(s) for s in my_sentences])
    print ('Maximum sequence length:', max_seq_len)
    # longest_sent = my_sentences[np.argmax(np.array([len(s) for s in my_sentences]))]
    # print (" ".join([word for word,_ in longest_sent]))
    ########################### HIST snetence lengths ############################
    # plt.hist([len(s) for s in my_sentences], bins=50)
    # plt.show()

    corpus = [[str(w[0]) for w in s] for s in my_sentences]
    tags = [[w[1] for w in s] for s in my_sentences]
    corpus_train, corpus_test, tags_train, tags_test = train_test_split(corpus, tags, test_size=0.2, random_state=101)
    
    
    #dump test data
    test_texts = [' '.join(seq) for seq in corpus_test]
    test_tags = [' '.join(seq) for seq in tags_test]

    with open(test_corpus_path, "w", encoding = "utf8") as f: #ISO-8859-1
        f.write('\n'.join(test_texts))
        print (f'test corpus contains {len(test_texts)} sequences')
    with open(test_tags_path, "w", encoding = "utf8") as f:
        f.write('\n'.join(test_tags))


    PAD_WORD = "ENDPAD"
    words = [str(w) for w in set([w[0] for s in my_sentences for w  in s])]
    words.append(PAD_WORD)
    words = sorted(words)
    n_words = len(words)
    print (f'vocabulary size is {n_words}')
    tags = [str(tag) for tag in set([w[1] for s in my_sentences for w  in s])]
    tags = sorted(tags)
    n_tags = len(tags)
    print (f'the different tags are {tags}')

    #########################  Converting words to numbers and numbers to words ###########
    # word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    with open(tags_dic_path, 'w') as tags_file:
      tags_file.write(json.dumps(tag2idx, sort_keys=True, indent=2))


    ### model config
    config_dic = {
        "max_seq_len":max_seq_len,
        "PAD_WORD":"ENDPAD",
        "PAD_TAG":"O"}
    
    with open(model_config_path, 'w') as f:
      f.write(json.dumps(config_dic, sort_keys=True, indent=2))

    # X = [[word2idx[w[0]] for w in s] for s in my_sentences]
    X = [[w[0] for w in s] for s in my_sentences]
    y = [[tag2idx[w[1]] for w in s] for s in my_sentences]
    # print (X[0])

    def pad_word_sequence (Corpus):
      padded_corpus = []
      for seq in Corpus:
        new_seq = []
        for i in range(max_seq_len):
          if i < len(seq):
            new_seq.append(seq[i])
          else:
            new_seq.append(PAD_WORD)
        padded_corpus.append(new_seq)
      return padded_corpus

    if bTrainMode:
        # X_train = [[word2idx[w] for w in s] for s in corpus_train]
        y_train = [[tag2idx[t] for t in s] for s in tags_train]
        # padding
        X_train = pad_word_sequence(corpus_train)
        y_train = pad_sequences(maxlen=max_seq_len, sequences=y_train, padding="post", value=tag2idx["O"])
        # sample_weights_train = np.array([get_weights_from_lables(np_seq) for np_seq in y])
        y_train = np.array([to_categorical(i, num_classes=n_tags) for i in y_train])
        
        y_test = [[tag2idx[t] for t in s] for s in tags_test]
        # padding
        X_test = pad_word_sequence(corpus_test)
        y_test = pad_sequences(maxlen=max_seq_len, sequences=y_test, padding="post", value=tag2idx["O"])
        y_test = np.array([to_categorical(i, num_classes=n_tags) for i in y_test])
        
        return X_train, X_test, y_train, y_test, max_seq_len, n_words, n_tags

    else:

        y_test = [[tag2idx[t] for t in s] for s in tags_test]
        # padding
        X_test = pad_word_sequence(corpus_test)
        y_test = pad_sequences(maxlen=max_seq_len, sequences=y_test, padding="post", value=tag2idx["O"])
        y_test = np.array([to_categorical(i, num_classes=n_tags) for i in y_test])

        return  X_test, y_test, max_seq_len, tag2idx


def build_model(max_seq_len, num_tags, batch_size):
    
    elmo_model = hub.Module(spec="../hub_elmo/", trainable=True)
    def ElmoEmbedding(x):
        return elmo_model(inputs={
                                "tokens": tf.squeeze(tf.cast(x, tf.string)),
                                "sequence_len": tf.constant(batch_size*[max_seq_len])
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]
    
    embedding_dim = 1024
    input_text = Input(shape=(max_seq_len,), dtype=tf.string)
    embedding = Lambda(ElmoEmbedding, output_shape=( max_seq_len, embedding_dim))(input_text)

    x = Bidirectional(LSTM(units=128, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)

    # x_rnn = Bidirectional(LSTM(units=128, return_sequences=True,
    #                        recurrent_dropout=0.2, dropout=0.2))(x)
    # x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(num_tags, activation="softmax"))(x)
    model = Model(input_text, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print (model.summary())
    return model


def train_model():
    sess = tf.Session()
    K.set_session(sess)
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    batch_size = 64
    X_train, X_test, y_train, y_test, max_seq_len, n_words, n_tags, _, _ = process_data()
    

    X_train = X_train[:batch_size*(len(X_train)//batch_size)]
    # y_train = y_train[:batch_size*(y_train.shape[0]//batch_size),:]
    # y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_train = np.array(y_train[:batch_size*(len(y_train)//batch_size)])

    
    X_test = X_test[:batch_size*(len(X_test)//batch_size)]
    # y_test = y_test[:batch_size*(y_test.shape[0]//batch_size),:]
    # y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
    y_test = np.array(y_test[:batch_size*(len(y_test)//batch_size)])

    model = build_model(max_seq_len=max_seq_len, num_tags=n_tags, batch_size=batch_size)


    # saver = tf.train.Saver()
    # with tf.Session() as sess:


    # Create a callback that saves the model's weights
    # keras_ckpt_path = './models/tf_session_model/tf_elmo_biLSTM_GMB_NER.ckpt'
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=keras_ckpt_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1)

    history = model.fit(np.array(X_train), y_train, validation_data=(np.array(X_test), y_test), batch_size=batch_size, 
                        epochs=1, validation_split=0.2, verbose=1)

    # json_config = model.to_json()
    # with open('./models/keras_config.json', 'w') as json_file:
    #     json_file.write(json_config)
    # model.sample_weights('./models/keras_weights.h5')
    
    model.save_weights(trained_model_path)
    # saver.save(sess, trained_tfsaver_path)
    # import pickle
    # pickle.dump(model, open(trained_model_path, 'wb'))


def evaluate_model ():


    batch_size = 64
    X_train, X_test, y_train, y_test, max_seq_len, n_words, n_tags, _, _ = process_data()
    

    X_train = X_train[:batch_size*(len(X_train)//batch_size)]
    # y_train = y_train[:batch_size*(y_train.shape[0]//batch_size),:]
    # y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
    y_train = np.array(y_train[:batch_size*(len(y_train)//batch_size)])

    
    X_test = X_test[:batch_size*(len(X_test)//batch_size)]
    # y_test = y_test[:batch_size*(y_test.shape[0]//batch_size),:]
    # y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
    y_test = np.array(y_test[:batch_size*(len(y_test)//batch_size)])

    model = build_model(max_seq_len=max_seq_len, num_tags=n_tags, batch_size=batch_size)

    # Reload the model from the 2 files we saved
    # with open('model_config.json') as json_file:
    #     json_config = json_file.read()
    # new_model = keras.models.model_from_json(json_config)
    model.load_weights(trained_model_path)

    preds = model.predict(np.array(X_test))
    preds = np.argmax(preds, axis=-1).flatten()
    y_true = np.argmax(y_test, axis=-1).flatten()

# ,target_names = list(tag2idx.keys())))
    print (classification_report(y_true,preds))



if __name__=='__main__':
    # X_train, X_test, y_train, y_test, max_seq_len, n_words, n_tags, _, _ = process_data()
    # build_model(140, 18, 64)
    # train_model()
    # evaluate_model()
    # process_data(bTrainMode=True, sub_sample_ratio=0.1)

    sample = '''Malta police arrested one of the country's most prominent businessmen on Wednesday in connection with an investigation into the murder of journalist Daphne Caruana Galizia.
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
The arrest was carried out on Thursday morning during a Europol-backed operation into money laundering, the statement from the Prime Minister's Office confirmed.'''
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(sample)
    print (len(sents))
    print ([len(sent.split()) for sent in sents])
    print (sents[5])
    new_sents = [(lambda sent: sent.split("\n") if len(sent.split())>50 else [sent])(sent) for sent in sents]
    new_sents = [sent for sent_list in new_sents for sent in sent_list]
    print (len(new_sents))
    print ([len(sent.split()) for sent in new_sents])
    # for sent in sents:
    #     print (sent)
    #     print ('new sentence :')
    # sents2 = sample.split("\n")
    # print (len(sents2))
    # print ([len(sent.split()) for sent in sents2])
    paragraph= '''An Australian academic freed by the Taliban in a prisoner swap has spoken of his "long and tortuous ordeal" as a hostage in Afghanistan.

Timothy Weeks said he believed US special forces had tried six times to rescue him and an American captive, Kevin King, who was also released.

Mr Weeks said he did not hate the Taliban, saying some of his guards were "lovely people" he hugged as he left.

"I never, ever gave up hope... I knew I would leave eventually," he said.

Mr Weeks and Mr King, also an academic, were freed this month in exchange for three senior militants held by the Afghan authorities, in a deal aimed at kick-starting peace talks.

The pair had been held for three years after being abducted outside the American University of Afghanistan in Kabul, where they worked as professors.'''
    
#     print (paragraph.split("\n"))
#     paragraph= '''hello there.
# there are folks.'''
    sents = sent_tokenize(paragraph)
    print (sum([ len(sent) for sent in sents]))
    print (len(paragraph))
    print (paragraph.index(sents[4]))
    print (paragraph[426:433])
    print (sents)
    sent1 = """the problem in the kitesurf strategy is 
to make the board go up the stream"""
    sent2 = "the kitesurf is to make the board go the stream"

    


