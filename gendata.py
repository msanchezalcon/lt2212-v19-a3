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





def preprocessing_data(filepath):
    """
    Takes file, removes punctuation and divides it into tokens
    Stores tokens in a sorted vocabulary list
    """
    vocab = []
    with open(filepath,"r",encoding="utf8") as f:
        file = f.read()
        clean_text = re.sub(r'([^\w\s\']|\n)', '', file)
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



def split_data(file, startline, endline):
    """
    Splits text (tokens) into train and test data and stores them into two separate lists.
    We first define a chunk of text we will use and that chunk will be separated into train and test every time.
    Always taking 80% randomly.
    """
    with open(file,"r",encoding="utf8") as f:
        file = f.read()
        clean_text = re.sub(r'([^\w\s\']|\n)', '', file)
        sent_token = sent_tokenize(clean_text)
        if startline is not None:
            sent_token = sent_token[startline:]
        if endline is not None:
            sent_token = sent_token[:endline]
    for sentence in sent_token:
        tokens = word_tokenize(sentence)

    train_split = 0.80
    train_data, test_data = tokens[:int(train_split * len(tokens))], tokens[int(train_split * len(tokens)):]
    train_data = shuffle(train_data) # randomizes selection every time
    test_data = shuffle(test_data)

    print(tokens)
    return train_data, test_data



def ngrams_builder(file, n, startline, endline):
    """
    Takes a list of tokens from the vocabulary list and returns n-grams as a list of tuples.
    """
    #with open(file, "r", encoding="utf8") as f:
       # file = f.read()
       # clean_text = re.sub(r"/[^\s]+", "", file)
        #clean_text = word_tokenize(clean_text)
    data = split_data(file, startline, endline)
    n_grams = ngrams(data, n, pad_left=True, left_pad_symbol="<s>")
    #print(n_grams)
    return list(n_grams)



def vector_builder(ngrams, vectordict):
    """
    Takes one-hot vectors from tokens and creates n-gram vectors (list of lists). Then convert that array into
    a dataframe.
    """
    main_vectors = []
    for gram in ngrams: # each token in the n-gram
        sub_vectors = []
        for word in gram[:-1]:
            sub_vectors += vectordict[word]
        sub_vectors.append(gram[-1])
        main_vectors.append(sub_vectors)
    array_of_vectors = np.array(main_vectors)
    dataframe = pd.DataFrame(array_of_vectors)

    return dataframe


def file_builder(outputfile):
    """
    Returns a csv version of the dataframe containing a representation of the array of vectors.
    """
    dataframe = vector_builder(ngrams, vectordict)

    return pd.DataFrame(dataframe).to_csv(outputfile)







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

vocabdict = preprocessing_data(args.inputfile)
vectordict = vector_builder(args.inputfile)
ngrams = ngrams_builder(args.inputfile, args.ngrams, args.startline, args.endline)
vector_builder(ngrams, vectordict)
split_data(args.inputfile, args.startline, args.endline)
ngrams_builder(args.inputfile, args.ngram, args.startline, args.endline)
file_builder(args.outputfile)