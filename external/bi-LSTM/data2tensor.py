import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize


class Mapper:
    def __init__(self):
        """
        Initialization of class Mapper
        Mapper is 
        """
        self.map = dict()
        self.map["GO"] = 0
        self.revmap = dict()
        self.revmap[0] = "GO"
        self.counter = 1
        self.review_max_words = 100
        self.summary_max_words = 100
        self.rev_sum_pair = None
        self.review_tensor = None
        self.summary_tensor = None
        self.review_tensor_reverse = None

    def generate_vocabulary(self, review_summary_file):
        """
        Get every word in each review and summary pair, and get rid of the punctuations
        review_summary_file: the same external variable in the stacked_bidirectional
        """
        self.rev_sum_pair = pd.read_csv(review_summary_file, header=0).values

        for review,summary in self.rev_sum_pair:
            review_list = wordpunct_tokenize(review)
            summary_list = wordpunct_tokenize(summary)
            # Use wordpunct_tokenize to get rid of punctuations
            self.__add_list_to_dict(review_list)
            self.__add_list_to_dict(summary_list)
            # Use __add_list_to_dict(target_list) to collect words in certain text (review or summary)
        self.map[""] = len(self.map)
        self.revmap[len(self.map)] = ""
        # Use the "" empty string as the last word of the voacabulary

    def __add_list_to_dict(self, target_list):
        """
        Add new words into vocabulary list
        target_list: the target list saving vocabulary for 
        """
        for word in target_list:
            word = word.lower()
            # Change capitalized letters into lower case letters
            if word not in self.map:
                # Add every new word to map and revmap in increasing order
                # Add one to counter when meeting new words
                # Counter is the size of the vocabulary
                self.map[word] = self.counter
                self.revmap[self.counter] = word
                self.counter += 1

    def get_tensor(self, reverseflag=False):
        """
        Generate input tensor for review or summary text
        reverseflag: boolean variable of generating tensor for review or reverse review
        """
        self.review_tensor = self.__generate_tensor(is_review=True)
        if reverseflag:
            # The tensor for reverse variable
            self.review_tensor_reverse = self.__generate_tensor(is_review=True, reverse=True)
        self.summary_tensor = self.__generate_tensor(is_review=False)

        if reverseflag:
            return self.review_tensor,self.review_tensor_reverse,self.summary_tensor
        else:
            return self.review_tensor, self.summary_tensor
        # Different return based on different reverse flag

    def __generate_tensor(self, is_review, reverse=False):
        """
        Generate 2 dimensional tensor for review or summary
        is_review: the boolean variable to judge generation for review or summary
        reverse: the boolean variable to distinguish the tensor for reverse review
        """
        seq_length = self.review_max_words if is_review else self.summary_max_words
        # Get the length of review or summary based on is_review
        total_rev_summary_pairs = self.rev_sum_pair.shape[0]
        # The number of review-summary combinations
        data_tensor = np.zeros([total_rev_summary_pairs,seq_length])
        # Initialize data_tensor

        sample = self.rev_sum_pair[0::, 0] if is_review else self.rev_sum_pair[0::, 1]
        # Update data_tensor through update sample
        # The first in rev_sum_pair is index, and the second is the entry point
        for index, entry in enumerate(sample.tolist()):
            index_list = np.array([self.map[word.lower()] for word in wordpunct_tokenize(entry)])
            # Reverse and get from backward
            if reverse:
                index_list = index_list[::-1]
            # Pad the list
            if len(index_list) <= seq_length:
                index_list = np.lib.pad(index_list, (0,seq_length - index_list.size), 'constant', constant_values=(0, 0))
            else:
                index_list = index_list[0:seq_length]
            # Renew the data_tensor
            data_tensor[index] = index_list
        return data_tensor

    def get_seq_length(self):
        return self.review_max_words

    def get_vocabulary_size(self):
        return len(self.map)

    def get_reverse_map(self):
        return self.revmap
