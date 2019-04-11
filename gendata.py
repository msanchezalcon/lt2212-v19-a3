import os, sys
import glob
import argparse
import numpy as np
import pandas as pd


# gendata.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here. You may not use the
# scikit-learn OneHotEncoder, or any related automatic one-hot encoders.
from nltk import word_tokenize, sent_tokenize, ngrams
import re
from sklearn.utils import shuffle

# Step by step:
# abrir archivo y preprocesar
# tokenizar
# ngramas
# hot code
# dataframe y guardarlo


def preprocessing_data(filepath, start, end):
    """
    Takes file, removes punctuation and divides it into tokens
    Stores tokens in a sorted vocabulary list, mapping them to an index.
    """
    vocab = []

    with open(filepath, "r", encoding="utf8") as f:
        file = f.read()
        clean_text = re.sub(r'/[^\s]+', '', file)
        sentences = sent_tokenize(clean_text)
        if start is not None:
            sentences = sentences[start:]
        if end is not None:
            sentences = sentences[:end]
    for sentence in sentences:
        tokenized_sentences = word_tokenize(sentence)

        for token in tokenized_sentences:
            if token not in vocab:
                vocab.append(token)


    sorted_vocab = sorted(vocab)
    map_tokens = {token: index for index, token in enumerate(sorted_vocab)}
    map_tokens["<s>"] = len(sorted_vocab)
    print(map_tokens)
    return map_tokens


def onehot_vector_builder(map_tokens):
    """
    Creating one-hot encoding vectors (that represent each word) by taking the indexes from vocabulary dictionary
    created in previous function.
    """
    onehot_vector_dict = {}
    for token, index in map_tokens.items():
        onehot_vector = [0] * len(map_tokens)
        onehot_vector[index] = 1
        onehot_vector_dict[token] = onehot_vector
    return onehot_vector_dict



def split_data(dataframe):
    """
    Splits text (tokens) into train and test data and stores them into two separate lists.
    We first define a chunk of text we will use and that chunk will be separated into train and test every time.
    The selected percentage will be chosen through the -P parsing argument.
    """

    split_by = 80
    k = dataframe.shape[0]*split_by // 100 # fila donde quiero cortar
    train_data= dataframe.loc[:k,:]
    test_data = dataframe.loc[k+1:,:] # cortamos filas, no columnas
    return train_data, test_data




def ngrams_builder(filepath, n, start, end):
    """
    Takes a list of tokens from the vocabulary list and returns n-grams as a list of tuples.
    """
    with open(filepath, "r", encoding="utf8") as f:
        file = f.read()
        clean_text = re.sub(r'/[^\s]+', '', file)
        sentences = sent_tokenize(clean_text)
        if start is not None:
            sentences = sentences[start:]
        if end is not None:
            sentences = sentences[:end]
    for sentence in sentences:
        tokenized_sentences = word_tokenize(sentence)
    n_grams = ngrams(tokenized_sentences, n, pad_left=True, left_pad_symbol= "<s>")
    return n_grams


def ngram_vector_builder(n_grams, vectordict):
    """
    Takes one-hot vectors from tokens and creates n-gram vectors (list of lists). Then convert that array into
    a dataframe.
    """
    main_vectors = [] # lista con un vector por cada gram
    for gram in n_grams:  # each token in the n-gram
        sub_vector = []
        for word in gram[:-1]: # no pilla el ultimo porque necesitanos la palabra entera (label)
            sub_vector += vectordict[word]
        sub_vector.append(gram[-1])
        main_vectors.append(sub_vector)
        # print(round(len(main_vectors)*100/len(n_grams), 2), "%")
    array_of_vectors = np.array(main_vectors)
    dataframe = pd.DataFrame(array_of_vectors)

    return dataframe







parser = argparse.ArgumentParser(description="Convert text to features")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3,
                    help="The length of ngram to be considered (default 3).")
parser.add_argument("-S", "--start", metavar="S", dest="startline", type=int,
                    default=None,
                    help="What line of the input data file to start from. Default is 0, the first line.")
parser.add_argument("-E", "--end", metavar="E", dest="endline",
                    type=int, default=None,
                    help="What line of the input data file to end on. Default is None, whatever the last line is.")
#parser.add_argument("-P", "--percentage", metavar="P", dest="percentage", type=float,
                    #default=0,
                    #help="What percentage of the data should be used.")
parser.add_argument("inputfile", type=str,
                    help="The file name containing the text data.")
parser.add_argument("outputfile", type=str,
                    help="The name of the output file for the feature table.")

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

# Global variables to be used again
vocabdict = preprocessing_data(args.inputfile, args.startline, args.endline) # map tokens
vectordict = onehot_vector_builder(vocabdict) # crea diccionario con key palabra y valor one hot
n_grams = ngrams_builder(args.inputfile, args.ngram, args.startline, args.endline)
all_hot_vectors_df = ngram_vector_builder(n_grams, vectordict)
train, test = split_data(all_hot_vectors_df)
train.to_csv(args.outputfile + "_train.txt")
test.to_csv(args.outputfile + "_test.txt", header=False)


