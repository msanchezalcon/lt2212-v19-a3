import os, sys
import glob
import argparse
import numpy as np
import pandas as pd


# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.
from nltk import word_tokenize
import re
from sklearn.utils import shuffle
from nltk import ngrams

parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3,
                    help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=0,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")




def preprocessing_data(file):
    """
    Takes file, removes punctuation and divides it into tokens
    Stores tokens in a sorted vocabulary list
    """
    vocab = []'0'
    with open(file,"r",encoding="utf8") as f:
        file = f.read()
        clean_text = re.sub(r"/[^\s]+", "", file)
        clean_text = word_tokenize(clean_text)
        for token in clean_text:
            if token not in vocab:
                vocab.append(token)

    sorted_vocab = sorted(vocab)
    map_tokens = {token: index for index, token in enumerate(sorted_vocab)}

    #print(map_tokens)
    return map_tokens



def vector_builder(inputfile):
    """
     Creating one-hot encoding vectors from vocabulary.
    """
    map_tokens = preprocessing_data(inputfile)
    onehot_vector_dict = {}
    for token, index in map_tokens.items():
        onehot_vector = [0] * len(map_tokens)
        onehot_vector[index] = 1
        onehot_vector_dict[token] = onehot_vector

    #print(one_hot_vector_dict)
    return onehot_vector_dict



def split_data(file):
    """
    Splits tokens into train and test data and stores them into two separate lists.
    """
    with open(file,"r",encoding="utf8") as f:
        file = f.read()
        clean_text = re.sub(r"/[^\s]+", "", file)
        clean_text = word_tokenize(clean_text)
        train_split = 0.80
        train_data, test_data = clean_text[:int(train_split * len(clean_text))], clean_text[int(train_split * len(clean_text)):]
        train_data = shuffle(train_data)
        test_data = shuffle(test_data)

        return train_data, test_data


def ngrams_builder(file, n):
    """
    Takes a list of tokens from the vocabulary list and returns n-grams as a list of tuples.
    """
    #with open(file, "r", encoding="utf8") as f:
       # file = f.read()
       # clean_text = re.sub(r"/[^\s]+", "", file)
        #clean_text = word_tokenize(clean_text)
    train_data, test_data = split_data(file)
    n_grams = ngrams(train_data, n, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="<e>")
    print(list(n_grams))
    return list(n_grams)


def n_gram_vector():
    """
    Takes one-hot vectors from tokens and creates n-gram vectors (list of lists). Then convert that array into
     a dataframe.
    """






preprocessing_data(args.inputfile)
split_data(args.inputfile)
ngrams_builder(args.inputfile, args.ngram)
vector_builder(args.inputfile)

args = parser.parse_args()

print("Loading data from file {}.".format(args.inputfile))
print("Starting from line {}.".format(args.startline))
if args.endline:
    print("Ending at line {}.".format(args.endline))
else:
    print("Ending at last line of file.")

print("Constructing {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.outputfile))

# THERE ARE SOME CORNER CASES YOU HAVE TO DEAL WITH GIVEN THE INPUT
# PARAMETERS BY ANALYZING THE POSSIBLE ERROR CONDITIONS.