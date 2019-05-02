import os, sys
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# test.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

def read_file(datafile):
    """
    Opens csv file created in gendata.py without the first column.
    """
    df = pd.read_csv(datafile, header=None)
    processed_csv = df.drop(df.columns[0], axis=1)
    return processed_csv


def open_model(modelfile):
    """
    Opens trained model.
    """
    f = open(modelfile, 'rb')
    model = pickle.load(f)
    f.close()

    return model


def vectors_labels(dataframe):
    """
    Gets the vectors (columns, except last one containing the labels and  class labels.
    """
    vectors = dataframe.iloc[:, :-1]
    labels = dataframe.iloc[:, -1]

    return vectors, labels
  
  
  # IN PROGRESS !!!

parser = argparse.ArgumentParser(description="Test a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features in the test data.")
parser.add_argument("modelfile", type=str,
                    help="The name of the saved model file.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
print("Loading model from file {}.".format(args.modelfile))

print("Testing {}-gram model.".format(args.ngram))

print("Accuracy is ...")
print("Perplexity is...")
