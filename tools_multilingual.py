"""
A selection of functions for encoding images and sentences
"""
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import numpy

from collections import OrderedDict, defaultdict
from scipy.linalg import norm

from utils import load_params, init_tparams
from model_multilingual import init_params, build_sentence_encoders, build_image_encoder
#from layers import shared_dropout_layer, get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer

def load_model(path_to_model=None, path_to_model_options=None,
        path_to_dictionaries=None, verbose=True):
    """
    Load all model components
    """
    if verbose:
        print path_to_model

    if path_to_model_options is None:
        path_to_model_options = "%s.pkl"%path_to_model

    # Load model options
    if verbose:
        print 'Loading model options...'

    with open(path_to_model_options, 'rb') as f:
        options = pkl.load(f)

    langs = options['langs']
    worddicts  = []
    iworddicts = []
    if verbose:
        print 'Loading dictionaries and creating inverted dictionaries...'

    path_to_dictionary = []
    if path_to_dictionaries is None:
        for lang in langs:
            path_to_dictionary.append("%s.dictionary-%s.pkl"%(path_to_model,lang))
    else:
        path_to_dictionary = path_to_dictionaries

    for lang_idx,lang in enumerate(langs):
        # Load the worddict
        with open(path_to_dictionary[lang_idx], 'rb') as f:
            worddict_lang = pkl.load(f)

        # Create inverted dictionaries
        word_idict_lang = dict()
        for kk, vv in worddict_lang.iteritems():
            word_idict_lang[vv] = kk
        word_idict_lang[0] = '<eos>'
        word_idict_lang[1] = 'UNK'

        worddicts.append(worddict_lang)
        iworddicts.append(word_idict_lang)

    # Load parameters
    if verbose:
        print 'Loading model parameters...'
    params = init_params(options)

    params = load_params(path_to_model, params)
    tparams = init_tparams(params)

    # Extractor functions
    if verbose:
        print 'Compiling sentence encoders...'
    trng = RandomStreams(1234)
    trng, in_outs = build_sentence_encoders(tparams, options)
    #trng, [xs, xmasks, sentences_all] = build_sentence_encoders(tparams, options)
    f_senc_all = []
    for i,lang in enumerate(langs):
        #x, xmask, sents = xs[i], xmasks[i], sentences_all[i]
        [x, xmask], sents = in_outs[i]
        f_senc_lang = theano.function([x, xmask], sents, name='f_senc_%s'%lang)
        f_senc_all.append(f_senc_lang)

    if verbose:
        print 'Compiling image encoder...'
    trng, [im], images = build_image_encoder(tparams, options)
    f_ienc = theano.function([im], images, name='f_ienc')

    # Store everything we need in a dictionary
    if verbose:
        print 'Packing up...'
    model = {}
    model['options'] = options
    model['f_ienc'] = f_ienc
    # TODO fix this tparams/params everywhere else in the rest of the code
    model['tparams'] = params

    for lang_idx, lang in enumerate(langs):
        model['worddict_%s'%lang]  =  worddicts[lang_idx]
        model['iworddict_%s'%lang] = iworddicts[lang_idx]
        model['f_senc_%s'%lang]    = f_senc_all[lang_idx]
    return model

def encode_multilingual_sentences(model, X, verbose=False, batch_size=128, lang='en'):
    """
    Encode sentences (specifically from a given language) into the joint embedding space
    """
    features = numpy.zeros((len(X), model['options']['dim_multimodal']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i,s in enumerate(captions):
        ds[len(s)].append(i)

    # quick check if a word is in the dictionary
    d = defaultdict(lambda : 0)
    for w in model['worddict_%s'%lang].keys():
        d[w] = 1

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]
            caption = [captions[c] for c in caps]

            seqs = []
            for i, cc in enumerate(caption):
                seqs.append([model['worddict_%s'%lang][w] if d[w] > 0 and model['worddict_%s'%lang][w] < model['options']['n_words_%s'%lang] else 1 for w in cc])
            x = numpy.zeros((k+1, len(caption))).astype('int64')
            x_mask = numpy.zeros((k+1, len(caption))).astype('float32')
            for idx, s in enumerate(seqs):
                x[:k,idx] = s
                x_mask[:k+1,idx] = 1.
            
            ff = model['f_senc_%s'%lang](x, x_mask)
            for ind, c in enumerate(caps):
                features[c] = ff[ind]

    return features


def encode_images(model, IM):
    """
    Encode images into the joint embedding space
    """
    images = model['f_ienc'](IM.astype('float32'))
    return images

