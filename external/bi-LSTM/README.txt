---------------------------------README-----------------------------------------
This source code is from https://github.com/harpribot/deep-summarization, which combines seq2seq models with every type of RNN

The code is annotated for better reading, however I havn't improved or rewritten much of them.

-----I will do this in spring break, and finish this part before the term continues.------

--------------------------------REQUIREMENTS-------------------------------------

matplotlib,nltk,numpy,pandas,pytest,scipy,tensorflow

----------------------------------FILES-----------------------------------------

The .py files have different use:

metric.py, bleu.py, bleu_scorer.py and rouge.py are for evaluating the summaries.

data2tensor.py creates the tensor for input data, which is based on the CSV files

checkpoint.py creates savings for NN during iterations

lstm_stacked_bidirectional.py, stacked_bidirectional.py, sequenceNet.py are the main structure of this program
They build a bidirectional LSTM using attention models for encoder and decoder.

train_script_lstm_stacked _bidirectional.py is the training file for the bi-directional LSTM

The most important file of all these files is stacked_bidirectional.py
