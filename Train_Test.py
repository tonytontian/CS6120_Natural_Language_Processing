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
from LSTM_model import model

# read data from json file and text file
def load_data(file_path):
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

# split data into three sections
# ratio of three kinds of data train:test:eval=8:1:1
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
    id2word_vec=[]
    word2id={}
    temp=1
    for i in word_vec:
        id2word.append(i)
        word2id[i]=temp
        temp+=1
        id2word_vec.append(word_vec[i])
    # append special tokens into the list
    # the length of word vector is 50
    PAD_vec=[0]*50
    GO_vec=[1]*50
    EOS_vec=[2]*50
    
    # add elements to word_list and vector_list in order
    id2word.append("<GO>")
    id2word_vec.append(GO_vec)
    word2id["<GO>"]=temp
    temp+=1
    id2word.append("<EOS>")
    id2word_vec.append(EOS_vec)
    word2id["<EOS>"]=temp
    temp+=1
    id2word.insert(0,"<PAD>")
    id2word_vec.insert(0,PAD_vec)
    word2id["<PAD>"]=0
    # generate numpy array for the wordvec
    vectors_array=np.array([np.array(xi) for xi in id2word_vec])
    return id2word,word2id,id2word_vec,vectors_array

# generate the ids of texts or summaries
def generate_ids(input_data,label):
    if label=="text":
        text_ids=np.zeros(shape=(batch_size,max_text_len),dtype='int32')
        i=0
        for text in input_data:
            text_word_list=text.split()
            j=0
            for word in text_word_list:
                try:
                    word_index=word2id[word]
                except KeyError:
                    word_index=word2id["unk"]
                text_ids[i,j]=word_index
                j+=1
                if j>=max_text_len:
                    break
            i+=1
        return text_ids
    elif label=="summary":
        summary_ids=np.zeros(shape=(batch_size,max_sum_len),dtype='int32')
        i=0
        for text in input_data:
            text_word_list=text.split()
            j=0
            for word in text_word_list:
                try:
                    word_index=word2id[word]
                except KeyError:
                    word_index=word2id["unk"]
                summary_ids[i,j]=word_index
                j+=1
                if j>=max_text_len:
                    break
            i+=1
        return summary_ids

# generate the feeding batch for feed_dicts
def yield_batch(embed_model,input_dataframe,batch_size):
    # yield batch for model training and predicting
    if embed_model=='self_trained':
        print('good')
        ###############
    elif embed_model=='word2vec':
        text_dataframe=input_dataframe["text"]
        summary_dataframe=input_dataframe["summary"]
        for i in range(0,len(text_dataframe)-len(text_dataframe) % batch_size,batch_size):
            text_batch=text_dataframe[i:i+batch_size-1]
            text_batch_len=[]
            for text in text_batch:
                text_batch_len.append(len(text.split(' ')))
                text_batch_ids=generate_ids(input_data=text_batch,label="text")
            text_batch_len=np.array(text_batch_len)
            summary_batch=summary_dataframe[i:i+batch_size]
            sum_batch_len=[]
            for summary in summary_batch:
                sum_batch_len.append(len(summary.split(' ')))
                sum_batch_ids=generate_ids(input_data=summary_batch,label="summary")
            sum_batch_len=np.array(sum_batch_len)
            yield (text_batch_ids,
                   sum_batch_ids,
                   text_batch_len,
                   sum_batch_len)


'''
MAIN FUNCTION
'''

file_path = os.getcwd()+"/idebate.json"
glove_path=os.getcwd()+"/glove.6B.50d.txt"
dataframe=load_data(file_path)

# initialize model parameters
epoch=10
learning_rate=0.001
text_embed_size=50
sum_embed_size=50
num_layers=1
attention='basic'
embed_model='word2vec'
batch_size=64
gradient_clip=0.9
max_text_len=400
max_sum_len=50

# preprocessing data under different mode
if embed_model=='self_trained':
    print('good')
    ###############
elif embed_model=='word2vec':
    word_vec=load_glove_model(glove_path)
    id2word,word2id,id2word_vec,vectors_array=list_generation(word_vec)

# initialize model
bi_lstm=model(learning_rate=learning_rate,text_embed_size=text_embed_size,sum_embed_size=sum_embed_size,\
                word2id=word2id,num_layers=num_layers,attention=attention,embed_model=embed_model,\
                batch_size=batch_size,gradient_clip=gradient_clip,pre_embed_model=vectors_array)

# train model and saver
with tf.Session() as sess:
    # start training the model with global initialization
    sess.run(tf.global_variables_initializer())
    # initialize the training saver as well
    print('Training starts.\n')
    # open a tensorboard file and a file wirter
    tensorboard_directory='tensorboard'
    if not os.path.exists(tensorboard_directory):
        os.makedirs(tensorboard_directory)
    merged_summary=tf.summary.scalar("loss",bi_lstm.loss)
    writer = tf.summary.FileWriter(tensorboard_directory)
    save_directory='checkpoints'
    saver=tf.train.Saver()
    if not os.path.exists(save_directory):
        os.makedirs(save_directory) 
    save_path=os.path.join(save_directory,'results')

    if embed_model=='self_trained':
        print('good')
    # the embedding model is word2vec
    elif embed_model=='word2vec':
        for a in range(1,epoch+1):
            print('Epoch %d/%d starts' % (a,epoch))
            train_dataframe,test_dataframe,eval_dataframe=data_split(dataframe)
            (train_text_batch,train_summary_batch,train_text_batch_len,train_sum_batch_len)=\
            next(yield_batch(embed_model=embed_model,input_dataframe=train_dataframe,batch_size=batch_size))

            for (train_text_batch,train_summary_batch,train_text_batch_len,train_sum_batch_len)\
                 in yield_batch(embed_model=embed_model,input_dataframe=train_dataframe,batch_size=batch_size):
                _,train_loss=sess.run(fetches=[bi_lstm.train_optimizer,bi_lstm.loss],feed_dict={\
                    bi_lstm.text:train_text_batch,
                    bi_lstm.summary:train_summary_batch,
                    bi_lstm.text_length:train_text_batch_len,
                    bi_lstm.sum_length:train_sum_batch_len,
                    })
            saver.save(sess=sess,save_path=save_path)
            # validation starts after training in every epoch
            time=0.0
            total_eval_loss=0.0
            print('Epoch %d/%d finishes training' % (a,epoch))
            (eval_text_batch,eval_summary_batch,eval_text_batch_len,eval_sum_batch_len)=\
            next(yield_batch(embed_model=embed_model,input_dataframe=eval_dataframe,batch_size=batch_size))

            for (eval_text_batch,eval_summary_batch,eval_text_batch_len,eval_sum_batch_len)\
                 in yield_batch(embed_model=embed_model,input_dataframe=eval_dataframe,batch_size=batch_size):
                eval_loss=sess.run(fetches=bi_lstm.loss,feed_dict={\
                    bi_lstm.text:eval_text_batch,
                    bi_lstm.summary:eval_summary_batch,
                    bi_lstm.text_length:eval_text_batch_len,
                    bi_lstm.sum_length:eval_sum_batch_len,
                    })
                total_eval_loss+=eval_loss
                time+=1
            print("Epoch %d/%d loss report: Train_loss: %.3f | Eval_loss: %.3f\n"\
                        % (a,epoch,train_loss,total_eval_loss/time))
    


