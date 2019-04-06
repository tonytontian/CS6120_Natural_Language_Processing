import tensorflow as tf
from stacked_bidirectional import StackedBidirectional

class LstmStackedBidirectional(StackedBidirectional):
    def __init__(self, review_summary_file, checkpointer, num_layers, attention=False):
        """
        Use the initialization of the StackedBidirectional
        External parameters are identical
        """
        super(LstmStackedBidirectional, self).__init__(review_summary_file, checkpointer, num_layers, attention)

    def get_cell(self):
        """
        Return the atomic RNN cell type used for this model
        """
        return tf.nn.rnn_cell.LSTMCell(self.memory_dim)
