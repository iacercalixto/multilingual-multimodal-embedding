# -*- coding:utf8 -*-
"""
Build multilingual dictionaries and save them to disk
"""
import cPickle as pkl
import os
import sys
import argparse

from vocab import build_dictionary
from datasets_multilingual import load_multilingual_dataset

def create_dictionaries(data, load_test, langs, dictionaries):
    train, dev, test = [], [], []
    for d,t in zip(data, load_test):
        train_, dev_, test_ = [], [], []
        for lang in langs:
            train_.append("%s/train.%s"%(d,lang))
            dev_.append("%s/dev.%s"%(d,lang))
            test_.append("%s/test.%s"%(d,lang))
        train.append(train_)
        dev.append(dev_)
        if t:
            test.append(test_)

    # debugging: make sure data set files exist
    assert( all([os.path.isfile(t) for t_ in train for t in t_]) ), "Could not find train files.\n%s"%train[0][0]
    assert( all([os.path.isfile(d) for d_ in dev for d in d_]) ), "Could not find dev files.\n%s"%dev[0][0]
    assert( all([os.path.isfile(t) for t_ in test for t in t_]) ), "Could not find test files.\n%s"%test[0][0]
    
    # Load training and development sets, alternatively also test set
    print 'Loading dataset'
    wordslang = []
    for d,t in zip(data, load_test):
        train, dev, test = load_multilingual_dataset(path_to_data=d,
                                                     langs=langs, load_test=t, load_images=False)
        for lidx,lang in enumerate(langs):
            #print len(train[0][lidx]), len(dev[0][lidx]), len(test[0][lidx])
            wordslang.append( train[0][lidx]+dev[0][lidx]+test[0][lidx] )

    worddicts = []
    iworddicts = []
    # Create and save dictionaries
    print 'Creating and saving multilingual dictionaries %s ...'%(", ".join(langs))
    for lidx, (lang, saveto) in enumerate(zip(langs,dictionaries)):
        worddict = build_dictionary(wordslang[lidx])[0]
        n_words_dict = len(worddict)
        print '%s dictionary size: %s'%(lang,str(n_words_dict))
        with open('%s.dictionary-%s.pkl'%(saveto,lang), 'wb') as f:
            pkl.dump(worddict, f)

    print 'Done.'

if __name__ == "__main__":
    # defaults
    # training data
    data_path='/media/storage2tb/resources/flickr30k/f30k-comparable-newsplits'
    load_test=[True, True]
    langs=['en', 'de']
    # output dictionaries
    dictionaries=['./dicts/f30k-example_dictionary.en',
                  './dicts/f30k-example_dictionary.de']
    verbose=True

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--data', default=data_path,
        help="""Full path to data set.""")
    parser.add_argument('--load-test', nargs="+", default=load_test, type='bool',
        help="""Load test set?""")
    parser.add_argument('--langs', nargs="+", default=langs,
        help="""Data set languages.""")
    parser.add_argument('--dictionaries', nargs="+", default=dictionaries,
        help="""Full path to dictionary prefix.""")
    parser.add_argument('--verbose', action='store_true', default=verbose)
    args = parser.parse_args()

    if args.verbose:
        print vars(args)

    # repeat data set for each language
    if not isinstance(args.data, list) or len(args.data) < len(args.langs):
        args.data = [args.data for _ in range(len(args.langs))]

    assert( len(args.data) == len(args.load_test) ==
            len(args.langs) == len(args.dictionaries) ),\
        'ERROR: Incorrect number of parameters!'

    create_dictionaries(args.data, args.load_test, args.langs, args.dictionaries)
