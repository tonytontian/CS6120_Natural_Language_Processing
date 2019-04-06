from seq2seq_ultimate import Seq2Seq
import sys
import os
import json
import nltk
import sys
import cchardet
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.layers import core as core_layers
from sklearn.cross_validation import train_test_split

if int(sys.version[0]) == 2:
    from io import open


# read data from json file and text file
def read_data(file_path):
    with open(file_path) as file:
        raw_data=file.read()
        json_data=json.loads(raw_data)
    dataframe=pd.DataFrame(columns=["text","summary"])
    for i in json_data:
        text_summary_pair={}
        text=" "
        for x in i["_argument_sentences"]:
            text=text+i["_argument_sentences"][x].lower()
            # probably we can get rid of stopping word and punctuaton for input
        text_summary_pair["text"]=text
        text_summary_pair["summary"]=i["_claim"]
        dataframe=dataframe.append(text_summary_pair,ignore_index=True)
        # append doesn't happen in place
    return dataframe

def data_split(dataframe):
    train_dataframe,testandvalidation_dataframe=train_test_split(dataframe,test_size=0.2)
    test_dataframe,eval_dataframe=train_test_split(testandvalidation_dataframe,test_size=0.5)
    return train_dataframe,test_dataframe,eval_dataframe

# open the word2vec file
def load_glove_model(glovefile):
    with open(glovefile,'rb') as detect_file:
            code_style=cchardet.detect(detect_file.read())
    f=open(glovefile,'r+',encoding=code_style['encoding'])
    word_vec={}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        word_vec[word] = embedding
    return word_vec

# generate a list of word_vec to words for word2vec mode
def list_generation(word_vec):
    id2word=[]
    id2wordvec=[]
    word2id={}
    temp=1
    for i in word_vec:
        id2word.append(i)
        word2id[i]=temp
        temp+=1
        id2wordvec.append(word_vec[i])
    # append special tokens into the list
    # the length of word vector is 50
    PAD_vec=[0]*50
    GO_vec=[1]*50
    EOS_vec=[2]*50
    
    # add elements to word_list and vector_list in order
    id2word.append("<GO>")
    id2wordvec.append(GO_vec)
    word2id["<GO>"]=temp
    temp+=1
    id2word.append("<EOS>")
    id2wordvec.append(EOS_vec)
    word2id["<EOS>"]=temp
    temp+=1
    id2word.insert(0,"<PAD>")
    id2wordvec.insert(0,PAD_vec)
    word2id["<PAD>"]=0
    # generate numpy array for the wordvec
    vectors_array=np.array([np.array(xi) for xi in id2wordvec])
    return id2word,word2id,id2wordvec,vectors_array

def preprocess_data():
    file_path = os.getcwd()+"/idebate.json"
    glove_path=os.getcwd()+"/glove.6B.50d.txt"
    dataframe = read_data(file_path)
    word_vec=load_glove_model(glove_path)

    id2word,word2id,id2wordvec,vectors_array=list_generation(word_vec)

    x_unk = word2id['unk']
    y_unk = word2id['unk']
    y_eos = word2id['<EOS>']

    X_indices = [[word2id.get(word, x_unk) for word in line.split(' ')] for line in dataframe["text"]]
    Y_indices = [[word2id.get(word, y_unk) for word in line.split(' ')] + [y_eos] for line in dataframe["summary"]]

    return X_indices, Y_indices, word2id, id2word, id2wordvec, vectors_array
# end function preprocess_data


def main():
    BATCH_SIZE = 64
    X_indices, Y_indices, word2id, id2word, id2wordvec, vectors_array = preprocess_data()
    X_train = X_indices[BATCH_SIZE:]
    Y_train = Y_indices[BATCH_SIZE:]
    X_test = X_indices[:BATCH_SIZE]
    Y_test = Y_indices[:BATCH_SIZE]

    model = Seq2Seq(
        rnn_size = 500,
        n_layers = 1,
        X_word2idx = word2id,
        encoder_embedding_dim = 50,
        Y_word2idx = word2id,
        decoder_embedding_dim = 50,
        pretrained_model=vectors_array
    )
    model.fit(X_train, Y_train, val_data=(X_test, Y_test), batch_size=BATCH_SIZE)
# end function main


if __name__ == '__main__':
    main()
