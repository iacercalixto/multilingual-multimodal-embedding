# -*- coding:utf8 -*-
"""
Main trainer function

This trains a model which includes translational evidence
(versus assuming all multilingual sentences are only comparable)
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import itertools

import time
import os
import warnings
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from vocab import build_dictionary
from evaluation_multilingual import i2t, t2i
from tools_multilingual import encode_multilingual_sentences, encode_images

from datasets_multilingual_with_translational_evidence \
        import load_multilingual_dataset
from model_multilingual_with_translational_evidence \
        import init_params, build_model, build_sentence_encoders, build_image_encoder
from homogeneous_data_multilingual_with_translational_evidence \
        import HomogeneousDataMultilingualWithTranslationalEvidence, prepare_data

# main trainer
def trainer(data=['f30k-comparable', 'f30k-translational'],
            langs=['en', 'de'],
            margin=1,
            dim=1600, # 800 forward, 800 backward
            dim_image=4096,
            dim_word=300,
            encoders={'en':'gru', 'de':'gru'}, # gru OR bow
            max_epochs=80,
            dispFreq=50,
            decay_c=0,
            grad_clip=2.,
            maxlen_w=100,
            optimizer='adam',
            batch_size = 128,
            saveto='./f30k-half-comparable-and-translational.npz',
            validFreq=100,
            lrate=0.0002,
            reload_=False,
            # new parameters
            minlen_w=10,
            max_words={'en':0, 'de':0}, # integer, zero means unlimited
            debug=False,
            use_dropout=True,
            dropout_prob=0.3,
            load_test=False,
            lambda_img_sent=0.75,
            lambda_sent_sent=0.25,
            bidirectional_enc=True,
            n_enc_hidden_layers=1):
            #use_all_costs=True):

    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['langs'] = langs
    for lang in langs:
        model_options['encoder_%s'%lang] = encoders[lang]
        model_options['max_words_%s'%lang] = max_words[lang]
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_image'] = dim_image
    model_options['dim_word'] = dim_word
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_
    model_options['minlen_w'] = minlen_w
    model_options['use_dropout'] = use_dropout
    model_options['dropout_prob'] = dropout_prob
    model_options['bidirectional_enc'] = bidirectional_enc
    model_options['n_enc_hidden_layers'] = n_enc_hidden_layers
    model_options['load_test'] = load_test
    model_options['lambda_img_sent'] = lambda_img_sent
    model_options['lambda_sent_sent'] = lambda_sent_sent
    #model_options['use_all_costs'] = use_all_costs

    assert(n_enc_hidden_layers>=1)

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    # Load training and development sets
    print 'Loading dataset'
    train, dev = load_multilingual_dataset(data, langs, load_test=load_test)[:2]

    # Create and save dictionaries
    print 'Creating and saving multilingual dictionaries %s'%(", ".join(model_options['langs']))
    worddicts = []
    iworddicts = []
    for lang_idx, lang in enumerate(langs):
        # built dictionaries including all comparable and translational vocab
        worddict = build_dictionary(train[0][0][lang_idx]+train[1][0][lang_idx]+
                                    dev[0][0][lang_idx]+dev[1][0][lang_idx])[0]
        n_words_dict = len(worddict)
        #print '%s dictionary size: %s'%(lang,str(n_words_dict))
        with open('%s.dictionary-%s.pkl'%(saveto,lang), 'wb') as f:
            pkl.dump(worddict, f)

        # Inverse dictionaries
        iworddict = dict()
        for kk, vv in worddict.iteritems():
            iworddict[vv] = kk
        iworddict[0] = '<eos>'
        iworddict[1] = 'UNK'

        worddicts.append(worddict)
        iworddicts.append(iworddict)

        model_options["n_words_%s"%lang] = n_words_dict if max_words[lang]==0 else max_words[lang]

    # assert all max_words per language are equal
    assert(all( x==max_words.values()[0] for x in max_words.values() ))

    print model_options

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, inps, is_translational, cost = build_model(tparams, model_options)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Building multilingual sentence encoders'
    trng, alls_se = build_sentence_encoders(tparams, model_options)
    f_sencs = []
    for inps_se in alls_se:
        #print "sentence encoder input", inps_se
        inp_se, sentences = inps_se
        f_senc = theano.function(inp_se, sentences, profile=False)
        f_sencs.append(f_senc)

    print 'Building image encoder'
    trng, inps_ie, images = build_image_encoder(tparams, model_options)
    f_ienc = theano.function(inps_ie, images, profile=False)

    print 'Building f_grad...',
    sys.stdout.flush()
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    sys.stdout.flush()
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print 'Optimization'

    # train:
    # [(c_train_caps_list, c_train_ims, 0), (t_train_caps_list, t_train_ims, 1)]

    # create training set iterator where
    # a heuristic tries to make sure sentences in minibatch have a similar size
    train_comparable_iter = HomogeneousDataMultilingualWithTranslationalEvidence(train[0],
                batch_size=batch_size, maxlen=maxlen_w, minlen=minlen_w)

    train_translational_iter = HomogeneousDataMultilingualWithTranslationalEvidence(train[1],
                batch_size=batch_size, maxlen=maxlen_w, minlen=minlen_w)

    uidx = 0
    curr = 0.
    curr_langs = [0.]*len(model_options['langs'])
    n_samples = 0

    ep_start = time.time()
    ep_times = [ ep_start ]
    for eidx in xrange(max_epochs):
        print 'Epoch ', eidx

        for xs, im, is_translational_ in itertools.chain(train_comparable_iter,train_translational_iter):
            uidx += 1
            xs, masks, im = prepare_data(xs, im, is_translational_, \
                                         worddicts, \
                                         model_options=model_options, \
                                         maxlen=maxlen_w)

            is_translational.set_value( is_translational_ )

            if xs[0] is None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            # do not train on certain small sentences (less than 3 words)
            #if not x_src.shape[0]>=minlen_w and x_tgt.shape[0]>= minlen_w:
            if not all( x.shape[0]>=minlen_w for x in xs ):
                print "At least one minibatch (in one of the languages in the model)",
                print "has less words than %i. Skipping..."%minlen_w
                skipped_samples += xs[0].shape[1]
                uidx -= 1
                continue

            n_samples += len(xs[0])

            # Update
            ud_start = time.time()
            # flatten inputs for theano function
            inps_ = []
            inps_.extend(xs)
            inps_.extend(masks)
            inps_.append(im)
            #inps_.append(is_translational_)
            #cost = f_grad_shared(xs, masks, im)
            cost = f_grad_shared(*inps_)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Translational ', is_translational_, 'UD ', ud

            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing results...'
                curr_model = {}
                curr_model['options'] = model_options
                # store model's language dependent parameters
                for lang, worddict, iworddict, f_senc in zip(langs, worddicts, iworddicts, f_sencs):
                    curr_model['worddict_%s'%lang] = worddict
                    curr_model['wordidict_%s'%lang] = iworddict
                    curr_model['f_senc_%s'%lang] = f_senc
                #curr_model['worddicts'] = worddicts
                #curr_model['wordidicts'] = iworddicts
                #curr_model['f_senc'] = f_senc
                curr_model['f_ienc'] = f_ienc

                # encode sentences
                lss = []
                for lang_idx, lang in enumerate(model_options['langs']):
                    # dev:
                    # ((c_dev_caps_list, c_dev_ims, 0), (t_dev_caps_list, t_dev_ims, 1))

                    dev_set = dev[ is_translational_ ]
                    #ls  = encode_multilingual_sentences(curr_model, dev[0][lang_idx], lang=lang)
                    ls  = encode_multilingual_sentences(curr_model, dev_set[0][lang_idx], lang=lang)
                    lss.append(ls)
                lim = encode_images(curr_model, dev_set[1].astype('float32'))
                #lim = encode_images(curr_model, dev[1].astype('float32'))

                # compute scores
                currscore = 0
                for i in range(len(lss)):
                    (r1, r5, r10, medr) = i2t(lim, lss[i])
                    print "Image to %s text: %.1f, %.1f, %.1f, %.1f" % (model_options['langs'][i], r1, r5, r10, medr)
                    (r1i, r5i, r10i, medri) = t2i(lim, lss[i])
                    print "%s text to image: %.1f, %.1f, %.1f, %.1f" % (model_options['langs'][i], r1i, r5i, r10i, medri)

                    # adjust current overall score
                    #currscore += r1 + r5 + r10 + r1i + r5i + r10i
                    currscore += r1 + (r5/1.5) + (r10/2) + r1i + (r5i/1.5) + (r10i/2)

                    # best current score for individual language/image pair
                    #currscore_lang = r1 + r5 + r10 + r1i + r5i + r10i
                    currscore_lang = r1 + (r5/1.5) + (r10/2) + r1i + (r5i/1.5) + (r10i/2)
                    if currscore_lang > curr_langs[i]:
                        curr_langs[i] = currscore_lang

                        # save model
                        print 'saving best %s...' % model_options['langs'][i],
                        params = unzip(tparams)
                        numpy.savez('%s.best-%s'%(saveto, model_options['langs'][i]), **params)
                        pkl.dump(model_options, open('%s.best-%s.pkl'%(saveto, model_options['langs'][i]), 'wb'))
                        print 'done'

                # adjust current best overall score if needed
                if currscore > curr:
                    curr = currscore

                    # Save model
                    print 'Saving best overall model (%s)...' % str("-".join(model_options['langs'])),
                    params = unzip(tparams)
                    numpy.savez(saveto, **params)
                    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                    print 'Done'

        ep_end = time.time()
        ep_times.append(ep_end)

        print 'Seen %d samples'%n_samples

    seconds = ep_times[-1] - ep_times[0]
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print "Finished execution in %d:%02d:%02d" % (h, m, s)

if __name__=="__main__":
    import argparse

    # prevent stdout buffering
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--saveto', type=str,
        default='/media/storage2tb/models/multilingual-multimodal'+\
                '/f30k-half-comparable-and-translational.npz')
    parser.add_argument('--margin', type=float, default=1)
    parser.add_argument('--max-epochs', type=int, default=80)
    parser.add_argument('--reload_', action='store_true')
    parser.add_argument('--use-dropout', action='store_true')
    parser.add_argument('--dropout-prob', type=float, default=0.3)
    parser.add_argument('--lambda-img-sent', type=float, default=0.75)
    parser.add_argument('--lambda-sent-sent', type=float, default=0.25)
    args = parser.parse_args()
    args = vars(args) # transform Namespace object into dict
    #print args
    #sys.exit()

    print "Starting..."
    trainer(**args)
