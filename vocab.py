"""
Constructing and loading dictionaries
"""
import cPickle as pkl
import numpy
from collections import OrderedDict

def build_dictionary(text, min_freq=0):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    min_freq: minimum frequency for a token to be included in the dictionary
              if token appears less than minimum frequency, it is discarded
              defaults to 0 (no minimum frequency, include all words)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1

    # if we require a minimum frequency for a word to be included in the dictionary
    if min_freq>0:
        wordcount = OrderedDict({k:v for k,v in wordcount.iteritems() if v>=min_freq})

    words = wordcount.keys()
    freqs = wordcount.values()
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    worddict[0] = '<eos>'
    worddict[1] = 'UNK'
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2 # 0: <eos>, 1: <unk>

    return worddict, wordcount

def load_dictionary(loc='./dict.pkl'):
    """
    Load a dictionary
    """
    with open(loc, 'rb') as f:
        worddict = pkl.load(f)
    return worddict

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'wb') as f:
        pkl.dump(worddict, f)
        pkl.dump(wordcount, f)


