import os
import json
import nltk
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.layers import core as core_layers
from sklearn.cross_validation import train_test_split

class model(object):
    def __init__(self,learning_rate,text_embed_size,sum_embed_size,word2id,\
                num_layers,attention,embed_model,batch_size,gradient_clip,pre_embed_model=None):
        # external hyperparameters
        # defined outside the class
        self.learning_rate=learning_rate
        # length of the embedded vector
        self.text_embed_size=text_embed_size
        self.sum_embed_size=sum_embed_size
        # list of word2index
        self.word2id=word2id
        # layers of the bi-LSTM cells
        self.num_layers=num_layers
        # select attention of the bi-LSTM network, possible values:'basic','B-attention','L-attention'
        self.attention=attention
        # choose the embedding model to process word vector,possible values:'word2vec','self_trained'
        self.embed_model=embed_model
        # size of batch for feed dict
        self.batch_size=batch_size
        # gradient clipping for optimization
        self.gradient_clip=gradient_clip
        # pre_embedding models like word2vec
        self.pre_embed_model=pre_embed_model
        

        # internal hyperparameters
        # number of node in one layer
        self.en_num_units=64
        self.de_num_units=64
        # define special tokens under self_training 
        if embed_model=='self_trained':
            '''
            
            '''

        # define special token under word2vec
        elif embed_model=='word2vec':
            self.token_GO=word2id["<GO>"]
            self.token_EOS=word2id["<EOS>"]
            self.token_PAD=word2id["<PAD>"]
            self.token_UNK=word2id["unk"]    
        
        # neural network definition and initialization
        # input layers, placeholder for one batch
        with tf.variable_scope('input'):
            # text and summary already converted from word to one-hot vector, but not into idvector
            if embed_model=='self_trained':
                # the basic int element means the onehot-vector for each word in each passage of one batch
                self.text=tf.placeholder(tf.int32,[None,None])
                self.summary=tf.placeholder(tf.int32,[None,None])
                # the basic int element means the number of words in each passage of one batch
                self.text_length=tf.placeholder(tf.int32,[None])
                self.sum_length=tf.placeholder(tf.int32,[None])

            # text and summary already converted from word to one-hot vector to wordvec
            elif embed_model=='word2vec':
                # the basic int element means wordvec(id) for each word in each passage of one batch
                self.text=tf.placeholder(tf.int32,[None,None])
                self.summary=tf.placeholder(tf.int32,[None,None])
                # the basic int element means the number of words in each passage of one batch
                self.text_length=tf.placeholder(tf.int32,[None])
                self.sum_length=tf.placeholder(tf.int32,[None])


        # encoder and decoder cells
        self.en_cell=tf.contrib.rnn.LSTMCell(num_units=(self.en_num_units)//2,initializer=tf.orthogonal_initializer(),reuse=False)
        self.de_cell=tf.contrib.rnn.LSTMCell(num_units=self.en_num_units,initializer=tf.orthogonal_initializer(),reuse=False)
        self.test_de_cell=tf.contrib.rnn.LSTMCell(num_units=self.en_num_units,initializer=tf.orthogonal_initializer(),reuse=True)

        # hidden layers for encoder
        with tf.variable_scope('encoder'):
            if embed_model=='self_trained':
                # obtain the id_vectors by training a converting matrix in the model itself
                # initialize the embedding matrix
                self.en_embedding=tf.get_variable(name='encoder_embedding',shape=[len(self.word2id),self.text_embed_size],\
                                            dtype=tf.float32,initializer=tf.random_uniform_initializer(-1.0, 1.0),trainable=true) 
                self.en_output=tf.nn.embedding_lookup(self.en_embedding,self.text)
            elif embed_model=='word2vec':
                # obtain the id_vectors by pre-trained model wod2vec
                # use the pre-embedded model for word-wordvec looking up
                self.en_pre_embed=tf.get_variable(name='encoder_pre_embedding',dtype=tf.float32,trainable=False,\
                                initializer=tf.constant(value=self.pre_embed_model,dtype=tf.float32))
                self.en_output=tf.nn.embedding_lookup(self.en_pre_embed,self.text)

            # use bidirectional RNN as the encoder structure
            for n in range(self.num_layers):
                (fwd_output,bkd_output),(fwd_state,bkd_state)=tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.en_cell,
                    cell_bw=self.en_cell,
                    inputs=self.en_output,
                    sequence_length=self.text_length,
                    dtype=tf.float32,
                    scope='biLSTM_'+str(n))
                self.en_output=tf.concat((fwd_output,bkd_output),2)
        
            # concentrate forward and backward state outputs
            # form one total_state tuple for decoder
            bi_state_c=tf.concat((fwd_state.c,bkd_state.c),-1)
            bi_state_h=tf.concat((fwd_state.h,bkd_state.h),-1)
            layer_state=tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c,h=bi_state_h)
            self.total_state=tuple([layer_state]*self.num_layers)

        # hidden layers for decoder
        # the first is constructed for training
        with tf.variable_scope('decoder'):
            if embed_model=='self_trained':
                # assume the same vocabulary size of summary and text
                self.de_embedding=tf.get_variable(name='decoder_embedding',shape=[len(self.word2id),self.sum_embed_size],\
                                        dtype=tf.float32,initializer=tf.random_uniform_initializer(-1.0,1.0))
                # remove last char in the form of one-hot vector
                main=tf.strided_slice(self.summary,[0,0],[self.batch_size,-1],[1,1])
                # add the one-hot vector of GO at the first position of sentence
                de_input=tf.concat(values=[tf.fill(dims=[self.batch_size,50],value=1),main],axis=1)
                de_input=tf.nn.embedding_lookup(self.de_embedding,de_input)
            elif embed_model=='word2vec':
                # obtain the id_vectors by pre-trained model wod2vec
                # use the pre-embedded model for word-wordvec looking up
                # remove last char in the form of id
                main=tf.strided_slice(self.summary,[0,0],[self.batch_size,-1],[1,1])
                # add wordvec of GO at the first position of sentence
                de_input=tf.concat(values=[tf.fill(dims=[self.batch_size,50],value=1),main],axis=1)
                self.de_pre_embed=tf.get_variable(name='decoder_pre_embedding',dtype=tf.float32,shape=[len(self.word2id),self.sum_embed_size])
                de_input=tf.nn.embedding_lookup(self.de_pre_embed,de_input)
            
            # define the structure of the de_cell based on attention
            if self.attention=='attention':
                # add attention to decoder
                self.attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(
                        num_units=self.de_num_units, 
                        memory=self.en_output,
                        memory_sequence_length=self.text_length)
                # wrap attention to encoder output
                self.decoder_cell=tf.contrib.seq2seq.AttentionWrapper(
                        cell=tf.nn.rnn_cell.MultiRNNCell([self.de_cell for _ in range(self.num_layers)]),
                        attention_mechanism=attention_mechanism,
                        attention_layer_size=self.de_num_units)

            elif self.attention=='basic':
                self.decoder_cell=tf.nn.rnn_cell.MultiRNNCell([self.de_cell for _ in range(self.num_layers)])

            # establish training helper
            training_helper=tf.contrib.seq2seq.TrainingHelper(
                inputs=de_input,
                sequence_length=self.sum_length,
                time_major=False)

            # use bidirectional RNN as decoder
            training_decoder=tf.contrib.seq2seq.BasicDecoder(
                cell=tf.nn.rnn_cell.MultiRNNCell([self.de_cell for _ in range(self.num_layers)]),
                helper=training_helper,
                initial_state=self.total_state,
                output_layer=tf.layers.Dense(len(self.word2id)))

            # dynamic decoder reduces time and space for calculation
            training_output,_,_=tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=tf.reduce_max(self.sum_length))
            
            # use basic decoder API
            # output=rnn_output,sample_id
            # rmm output can be used to calculate loss function
            self.training_logits=training_output.rnn_output
        
        # the second is constructed for predicting
        with tf.variable_scope('decoder',reuse=True):
            # define the structure of the de_cell based on attention
            if self.attention=='attention':
                # add attention to decoder
                self.attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(
                        num_units=self.de_num_units, 
                        memory=self.en_output,
                        memory_sequence_length=self.text_length)
                # wrap attention to encoder output
                self.decoder_cell=tf.contrib.seq2seq.AttentionWrapper(
                        cell=tf.nn.rnn_cell.MultiRNNCell([test_de_cell for _ in range(self.num_layers)]),
                        attention_mechanism=attention_mechanism,
                        attention_layer_size=self.de_num_units)

            elif self.attention=='basic':
                self.decoder_cell=tf.nn.rnn_cell.MultiRNNCell([self.test_de_cell for _ in range(self.num_layers)])

            # establish predicting helper
            if embed_model=='self_trained':
                predicting_helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=tf.get_variable('decoder_embedding'),
                    start_tokens=tf.tile(input=tf.constant([self.token_GO],dtype=tf.int32),multiples=[self.batch_size]),
                    end_token=self.token_EOS)
            elif embed_model=='word2vec':
                predicting_helper=tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=self.en_pre_embed,
                    start_tokens=tf.tile(input=tf.constant([self.token_GO],dtype=tf.int32),multiples=[self.batch_size]),
                    end_token=self.token_EOS)

            # use bidirectional RNN as decoder
            predicting_decoder=tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=predicting_helper,
                initial_state=self.total_state,
                output_layer=tf.layers.Dense(len(self.word2id),_reuse=True))

            # dynamic decoding
            predicting_output,_,_=tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished = True)

            # use basic decoder output API
            self.predicting_ids=predicting_output.sample_id
        
        with tf.variable_scope('loss_and_optimizer'):
            weights=tf.sequence_mask(maxlen=tf.reduce_max(self.sum_length),lengths=self.sum_length,dtype=tf.float32)
            self.loss=tf.contrib.seq2seq.sequence_loss(logits=self.training_logits,
                                                    targets=self.summary,
                                                    weights=weights)
            # gradient clipping
            params=tf.trainable_variables()
            gradients=tf.gradients(self.loss,params)
            clipped_gradients,_=tf.clip_by_global_norm(gradients,self.gradient_clip)
            self.train_optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(zip(clipped_gradients,params))
