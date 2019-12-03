import tensorflow_hub as hub
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, Lambda
from keras.layers.merge import add
import fasttext as ft
from sklearn.externals import joblib

from keras import backend as K
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from time import time

trained_model_path = './models/keras_ElMo_biLSTM_GMB_ner.h5'
trained_tfsaver_path = './models/tf_session_model/tf_elmo_biLSTM_GMB_NER'
SUB_SAMPLE_RATION=1.0
MAX_SEQUENCE_LENGTH = 256
MAX_NUM_WORDS = 40000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

class SentenceGetter(object):
    
    def __init__(self, dataset, SUB_SAMPLE_RATION):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        def filter_tags (tag):
            if str(tag)[2:] in ['art', 'eve', 'nat']:
                return 'O'
            else:
                return tag
        agg_func = lambda s: [(w, filter_tags(t)) for w,t in zip(s["word"].values.tolist(),
                                                        s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        ########################  SUBSAMPLE dataset #################################
        index = np.arange(0,self.grouped.shape[0])
        np.random.seed(101) 
        np.random.shuffle(index)
        index = index[0: int(SUB_SAMPLE_RATION*index.shape[0])]
        self.grouped = self.grouped.iloc[index].reset_index(drop=True)

        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def process_data(bTrainMode):
    raw_data_path = '../datasets/GMB_ner/'
    # print(check_output(["ls", raw_data_path]).decode("utf8"))

    # Any results you write to the current directory are saved as output.

    dframe = pd.read_csv(raw_data_path+'ner.csv', encoding = "ISO-8859-1", error_bad_lines=False)
    # print (dframe.head())
    # print (dframe.columns)
    dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word',"pos"],axis=1)
    
    dataset=dataset.drop(['shape'],axis=1)
    # print(dataset.head())


    getter = SentenceGetter(dataset, SUB_SAMPLE_RATION)
    my_sentences = getter.sentences
    print (f'corpus contains {len(my_sentences)} sentences')
    # print (my_sentences[0])
    max_seq_len = max([len(s) for s in my_sentences])
    print ('Maximum sequence length:', max_seq_len)
    # longest_sent = my_sentences[np.argmax(np.array([len(s) for s in my_sentences]))]
    # print (" ".join([word for word,_ in longest_sent]))
    ########################### HIST snetence lengths ############################
    # plt.hist([len(s) for s in my_sentences], bins=50)
    # plt.show()

    corpus = [[str(w[0]) for w in s] for s in my_sentences]
    tags = [[w[1] for w in s] for s in my_sentences]
    corpus_train, corpus_test, tags_train, tags_test = train_test_split(corpus, tags, test_size=VALIDATION_SPLIT, random_state=101)
    
    if bTrainMode:
        train_texts = [' '.join(seq) for seq in corpus_train]
        
        #fine tune fasttext
        with open('./temp_data/train_ft.txt', "w", encoding = "ISO-8859-1") as train_corpus_file:
            train_corpus_file.writelines(train_texts)

        t0 = time()
        print ('starting training of fasttext model...')
        # np.savetxt(fname='./temp_data/train_X.csv', X= np.array(train_texts), delimiter=',', fmt="%s")
        model = ft.train_unsupervised(input='./temp_data/train_ft.txt', model='skipgram', verbose=0, dim=EMBEDDING_DIM, epoch=10)
        model.save_model('./models/ft_model.bin')
        print (f'training of fastText done in {time() - t0}')


    
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
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    # print (f"idx of O is {tag2idx['O']}")

    if bTrainMode:
        X_train = [[word2idx[w] for w in s] for s in corpus_train]
        y_train = [[tag2idx[t] for t in s] for s in tags_train]
        # padding
        X_train = pad_sequences(maxlen=max_seq_len, sequences=X_train, padding="post",value=word2idx[PAD_WORD])
        y_train = pad_sequences(maxlen=max_seq_len, sequences=y_train, padding="post", value=tag2idx["O"])
        y_train = np.array([to_categorical(i, num_classes=n_tags) for i in y_train])
        
        return X_train, y_train, max_seq_len, word2idx, n_tags, tags

    else:
        X_test = [[word2idx[w] for w in s] for s in corpus_test]
        y_test = [[tag2idx[t] for t in s] for s in tags_test]
        # padding
        X_test = pad_sequences(maxlen=max_seq_len, sequences=X_test, padding="post",value=word2idx[PAD_WORD])
        y_test = pad_sequences(maxlen=max_seq_len, sequences=y_test, padding="post", value=tag2idx["O"])
        y_test = np.array([to_categorical(i, num_classes=n_tags) for i in y_test])

        return  X_test, y_test, max_seq_len, word2idx, n_tags, tags



def build_model(max_seq_len, num_tags, word2idx):
    # prepare embedding matrix
    # tokenizer_filename = './models/keras_tokenizer.sav'
    # tokenizer = joblib.load(tokenizer_filename)
    ft_model = ft.load_model('./models/ft_model.bin')

    # word_index = tokenizer.word_index
    num_words = min(MAX_NUM_WORDS, len(word2idx))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = ft_model.get_word_vector(word)
        assert embedding_vector is not None
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

    
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_seq_len,
                                trainable=True)

    sequence_input = Input(shape=(max_seq_len,), dtype='int32')
    embedded_sequences  = embedding_layer(sequence_input)

    x = Bidirectional(LSTM(units=128, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedded_sequences)

    # x_rnn = Bidirectional(LSTM(units=128, return_sequences=True,
    #                        recurrent_dropout=0.2, dropout=0.2))(x)
    # x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(num_tags, activation="softmax"))(x)
    model = Model(sequence_input, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print (model.summary())
    return model


def train_model():
    
    X_train, y_train, max_seq_len, word2idx, n_tags, _ = process_data(bTrainMode=True)

    model = build_model(max_seq_len=max_seq_len, num_tags=n_tags, word2idx=word2idx)

    batch_size = 128
    history = model.fit(X_train, y_train,  batch_size=batch_size, 
                        epochs=1, validation_split=0.2, verbose=1) #validation_data=(X_test, y_test),
    model.save(trained_model_path)


def evaluate_model ():

    X_test, y_test, max_seq_len, word2idx, n_tags, tags = process_data(bTrainMode=False)
    
    model = build_model(max_seq_len=max_seq_len, num_tags=n_tags, word2idx=word2idx)
    model.load_weights(trained_model_path)

    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=-1).flatten()
    y_true = np.argmax(y_test, axis=-1).flatten()
    # my_index = y_true!=16
    # y_true = y_true[my_index]
    # preds = preds[my_index]
    # tags.remove('O')
    print (classification_report(y_true,preds,target_names = tags))

    # Reload the model from the 2 files we saved






if __name__=='__main__':
    # X_train, X_test, y_train, y_test, max_seq_len, n_words, n_tags, _, _ = process_data()
    # fine_tune_fastext_model()
    # build_model(140, 18, 64)
    # tokenize_with_keras()
    # test_load_tokenizer()
    train_model()
    evaluate_model()



def fine_tune_ft_model ():
    model = ft.train_unsupervised(input='./temp_data/train_X.csv', model='skipgram', verbose=0, dim=100, epoch=10)
    model.save_model('./models/ft_model.bin')


def tokenize_with_keras ():
    f = open('./temp_data/train_X.csv', "r")
    texts = f.readlines()
    f.close()
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    # json_config = tokenizer.to_json()
    # with open('./models/keras_tokenizer_config.json', 'w') as json_file:
    #     json_file.write(json_config)

    tokenizer_filename = './models/keras_tokenizer.sav'
    joblib.dump(tokenizer, tokenizer_filename)

def test_load_tokenizer ():
    tokenizer_filename = './models/keras_tokenizer.sav'
    tokenizer = joblib.load(tokenizer_filename)
    # tokenizer = tf.keras.preprocessing.text.tokenizer_from_json()
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))