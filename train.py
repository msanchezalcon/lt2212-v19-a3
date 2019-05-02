import argparse
import numpy as np
import pandas as pd
import pickle
import os, sys
from sklearn.linear_model import LogisticRegression

# train.py -- Don't forget to put a reasonable amount code comments
# in so that we better understand what you're doing when we grade!

# add whatever additional imports you may need here.

def read_file(dataframe):
    """
    Opens csv file created in gendata.py without the first column.
    """
    df = pd.read_csv(dataframe, header=None)
    processed_csv = df.drop(df.columns[0], axis=1)
    return processed_csv

def train_model(dataframe):
    """
    Using logistic regression to train model taking x feature values and y target values.
    """
    x = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]
    train = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    model = train.fit(x, y)

    return model

def pickle(model):
    pickle.dump(model, open(args.modelfile, 'wb'))

parser = argparse.ArgumentParser(description="Train a maximum entropy model.")
parser.add_argument("-N", "--ngram", metavar="N", dest="ngram", type=int, default=3, help="The length of ngram to be considered (default 3).")
parser.add_argument("datafile", type=str,
                    help="The file name containing the features.")
parser.add_argument("modelfile", type=str,
                    help="The name of the file to which you write the trained model.")

args = parser.parse_args()

print("Loading data from file {}.".format(args.datafile))
print("Training {}-gram model.".format(args.ngram))
print("Writing table to {}.".format(args.modelfile))

dataframe = read_file(args.datafile)
model = train_model(dataframe)
pickle(model)

# YOU WILL HAVE TO FIGURE OUT SOME WAY TO INTERPRET THE FEATURES YOU CREATED.
# IT COULD INCLUDE CREATING AN EXTRA COMMAND-LINE ARGUMENT OR CLEVER COLUMN
# NAMES OR OTHER TRICKS. UP TO YOU.
