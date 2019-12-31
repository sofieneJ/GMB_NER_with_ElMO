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
from bert.tokenization import FullTokenizer
import json


test_corpus_path = './temp_data/test_copus.txt'
test_tags_path = './temp_data/test_tags.txt'
tags_dic_path = './models/tags_dic.json'
model_config_path = './models/model_config.json'
bert_path = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1" 
sub_sample_ratio=0.01


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run(
          [
              tokenization_info["vocab_file"],
              tokenization_info["do_lower_case"],
          ]
      )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

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

    np_test_data = np.array((test_texts, test_tags))

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

def relabel_after_tokenization(word_seq, word_tags_seq, token_seq):
  token_tags_seq = ['O' for _ in token_seq]
  sentence = ' '.join(word_seq)
  accumulated_offset = 0
  for i,token in enumerate(token_seq):
    if '##' not in token:
      min_search_start_index = sum([len(tok.strip('#')) for k,tok in enumerate(token_seq) if k<i])
      min_search_start_index += accumulated_offset
      my_index = sentence.index(token, min_search_start_index)
      accumulated_offset += my_index - min_search_start_index
      word_index = 0
      
      while len(' '.join(word_seq[0:word_index])) < my_index:
        word_index +=1
      
      if word_index>0:
        word_index-=1
      
      current_tag = word_tags_seq[word_index]
      token_tags_seq[i] = current_tag
      j = i+1
      while j<len(token_seq) and '##' in token_seq[j]:
        token_tags_seq[j] = 'X'
        j+=1
  
  return token_tags_seq

if __name__ == "__main__":
    # bert_tokenizer = create_tokenizer_from_hub_module()
    # #test the data function
    # _, _, _, _, _ = process_data(tokenizer=bert_tokenizer, bTrainMode=True, sub_sample_ratio=0.1, max_seq_len=256)

  word_seq = ['Hello', "I'm", 'the', 'OVH', 'globalization', 'worldchampion', 'my', 'friend', 'Victor', 'Chang!']
  word_tags_seq = ['O', 'O', 'O', 'B-org', 'O', 'O', 'O', 'O', 'B-per', 'I-per']
  token_seq = ['Hello', 'I', "'", 'm', 'the', 'O', '##V', '##H', 'global', '##ization', 'world', '##champ', '##ion', 'my', 'friend', 'Victor', 'Chang', '!']

  token_tags_seq = relabel_after_tokenization(word_seq, word_tags_seq, token_seq)
  print ([(tok,tag) for tok,tag in zip(token_seq,token_tags_seq)])