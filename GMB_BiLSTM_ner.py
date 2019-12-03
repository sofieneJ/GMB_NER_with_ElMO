# cf https://www.kaggle.com/navya098/bi-lstm-for-ner
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.layers.merge import add
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

trained_model_path = './models/keras_biLSTM_GMB_ner.h5'
test_corpus_path = './temp_data/test_copus.txt'
test_tags_path = './temp_data/test_tags.txt'
vocab_dic_path = './models/vocab_dic.json'
tags_dic_path = './models/tags_dic.json'
model_config_path = './models/model_config.json'

class SentenceGetter(object):
    
    def __init__(self, dataset, sub_sample_ratio):
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
    print(check_output(["ls", raw_data_path]).decode("utf8"))

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
    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    
    #########################  Dumping dictionaries and params #####################
    with open(vocab_dic_path, 'w') as vocab_file:
      vocab_file.write(json.dumps(word2idx, sort_keys=True, indent=2))
    with open(tags_dic_path, 'w') as tags_file:
      tags_file.write(json.dumps(tag2idx, sort_keys=True, indent=2))
    
    config_dic = {
        "max_seq_len":max_seq_len,
        "PAD_WORD":"ENDPAD",
        "PAD_TAG":"O"}
    
    with open(model_config_path, 'w') as f:
      f.write(json.dumps(config_dic, sort_keys=True, indent=2))

    if bTrainMode:
        X_train = [[word2idx[w] for w in s] for s in corpus_train]
        y_train = [[tag2idx[t] for t in s] for s in tags_train]
        # padding
        X_train = pad_sequences(maxlen=max_seq_len, sequences=X_train, padding="post",value=word2idx[PAD_WORD])
        y_train = pad_sequences(maxlen=max_seq_len, sequences=y_train, padding="post", value=tag2idx["O"])
        # sample_weights_train = np.array([get_weights_from_lables(np_seq) for np_seq in y])
        y_train = np.array([to_categorical(i, num_classes=n_tags) for i in y_train])
        
        X_test = [[word2idx[w] for w in s] for s in corpus_test]
        y_test = [[tag2idx[t] for t in s] for s in tags_test]
        # padding
        X_test = pad_sequences(maxlen=max_seq_len, sequences=X_test, padding="post",value=word2idx[PAD_WORD])
        y_test = pad_sequences(maxlen=max_seq_len, sequences=y_test, padding="post", value=tag2idx["O"])
        y_test = np.array([to_categorical(i, num_classes=n_tags) for i in y_test])
        
        return X_train, X_test, y_train, y_test, max_seq_len, n_words, n_tags

    else:
        X_test = [[word2idx[w] for w in s] for s in corpus_test]
        y_test = [[tag2idx[t] for t in s] for s in tags_test]
        # padding
        X_test = pad_sequences(maxlen=max_seq_len, sequences=X_test, padding="post",value=word2idx[PAD_WORD])
        y_test = pad_sequences(maxlen=max_seq_len, sequences=y_test, padding="post", value=tag2idx["O"])
        y_test = np.array([to_categorical(i, num_classes=n_tags) for i in y_test])

        return  X_test, y_test, max_seq_len, word2idx, tag2idx


def build_model(max_seq_len, vocab_size, num_tags):
    embedding_dim = 256
    input = Input(shape=(max_seq_len,))
    embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len)(input)
    #     embeddings = Dropout(0.1)(embeddings)
    x1 = Bidirectional(LSTM(units=128, return_sequences=True, recurrent_dropout=0.1))(embeddings)

    x2 = Bidirectional(LSTM(units=128, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x1)
    #     x = add([x1, x2])  # residual connection to the first biLSTM

    out = TimeDistributed(Dense(units=num_tags, activation="softmax"))(x2)  # softmax output layer

    model = Model(input, out)
    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])#, sample_weight_mode="temporal"

    print (model.summary())

    return model

def train_model(sub_sample_ratio):

    X_train, X_test, y_train, y_test, max_seq_len, n_words, n_tags = process_data(bTrainMode=True, sub_sample_ratio=sub_sample_ratio)
    model = build_model(max_seq_len=max_seq_len, vocab_size=n_words, num_tags=n_tags)
    
    # class_weights = {i:10.0 for i in range(n_tags)}
    # class_weights[tag2idx['O']] = 1.0

    history = model.fit(X_train, np.array(y_train), validation_data=(X_test, np.array(y_test)), 
        batch_size=64, epochs=1, validation_split=0.2, verbose=1) #, sample_weight=tr_sample_weights
    
    model.save(trained_model_path)

def evaluate_model(sub_sample_ratio):
    X_test, y_test, max_seq_len, word2idx, tag2idx = process_data(bTrainMode=False, sub_sample_ratio=sub_sample_ratio)
    model = build_model(max_seq_len=max_seq_len, vocab_size=len(word2idx), num_tags=len(tag2idx))

    model.load_weights(trained_model_path)

    preds = model.predict(np.array(X_test))
    preds = np.argmax(preds, axis=-1).flatten()
    y_true = np.argmax(y_test, axis=-1).flatten()

    print (classification_report(y_true,preds,target_names = list(tag2idx.keys())))

    # for i in range(0,10):
    #     p = model.predict(np.array([X_test[i]]))
    #     p = np.argmax(p, axis=-1)
    #     y_true = np.argmax(y_test[i], axis=-1)
    #     print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
    #     for w, true_tag_id, pred in zip(X_test[i],y_true, p[0]):
    #         if words[w]!='ENDPAD':
    #             print("{:14}: {:5} {}".format(words[w],tags[true_tag_id], tags[pred])) 

    # model.evaluate(x=X_test, y= np.array(y_test))

def test_inference(text_sequence, true_tags_seq=None):  
    ## loading the model
    with open(vocab_dic_path, 'r') as f:
        word2idx = json.load(f)
    with open(tags_dic_path, 'r') as f:
        tags2idx=json.load(f)
    with open(model_config_path, 'r') as f:
        config_dic=json.load(f)

    model = build_model(max_seq_len=config_dic["max_seq_len"], vocab_size=len(word2idx), num_tags=len(tags2idx))
    model.load_weights(trained_model_path)
                                    

    idx2word = {idx:word for word,idx in word2idx.items()}
    idx2tag = {idx:tag for tag,idx in tags2idx.items()} 
  
    # text_seq = test_texts[i].split()
    X_test = [[word2idx[w] for w in text_sequence]]
    # padding
    X_test = pad_sequences(maxlen=config_dic["max_seq_len"], sequences=X_test, padding="post",value=word2idx[config_dic["PAD_WORD"]])
    # print (X_test)
    p = model.predict(X_test)
    p = np.argmax(p, axis=-1)

    if true_tags_seq != None:
        y_test = [[tags2idx[t] for t in true_tags_seq]]
        y_test = pad_sequences(maxlen=config_dic["max_seq_len"], sequences=y_test, padding="post", value=tags2idx[config_dic["PAD_TAG"]])
        y_test = np.array([to_categorical(i, num_classes=len(tags2idx)) for i in y_test])
        # print (y_test)
        y_true = np.argmax(y_test, axis=-1)
        print("{:14} ({:5}): {}".format("Word", "True", "Pred"))
        for w, true_tag_id, pred in zip(X_test[0],y_true[0], p[0]):
            if idx2word[w]!='ENDPAD':
                print("{:14}: {:5} {}".format(idx2word[w],idx2tag[true_tag_id], idx2tag[pred])) 

    predicted_tags_seq = [idx2tag[pred]  for w, pred in zip(X_test[0], p[0])  if idx2word[w]!='ENDPAD']
    print (predicted_tags_seq)
    print(text_sequence)
    predicted_entities = []
    for i,entity_name in enumerate(predicted_tags_seq):
        if entity_name != config_dic["PAD_TAG"]:
            start_index = len(' '.join(text_sequence[:i]))
            entity_len = len(text_sequence[i])
            entity = {
                "entity":entity_name,
                "start_index":start_index,
                "entity_length":entity_len
            }
            predicted_entities.append(entity)

    print (json.dumps(predicted_entities, sort_keys=True, indent=2) )
    return predicted_entities
            
if __name__=='__main__':
    
    sub_sample_ratio=1.0
    # process_data(sub_sample_ratio)
    # train_model(sub_sample_ratio)
    # evaluate_model(sub_sample_ratio)
      ## loading the data      


    with open(test_corpus_path, "r", encoding = "utf8") as f:
        test_texts = f.readlines()
        print (f'test corpus contains {len(test_texts)} sequences')
    with open(test_tags_path, "r", encoding = "utf8") as f:
        test_tags = f.readlines()
    
    for i in range (0,1):
        test_inference(test_texts[i].split())#, test_tags_path.split()

    # batch_pad = np.array([["hello" for _ in range (0, 123)] for _ in range(0,63)])
    # print (batch_pad.shape)
    # print (batch_pad[1])