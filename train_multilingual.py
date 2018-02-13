# -*- coding:utf8 -*-
"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

import homogeneous_data_multilingual

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model_multilingual import init_params, build_model, build_sentence_encoders, build_image_encoder
from vocab import build_dictionary
from evaluation_multilingual import i2t, t2i, t2t
from tools_multilingual import encode_multilingual_sentences, encode_images, load_model
from datasets_multilingual import load_multilingual_dataset

# main trainer
def trainer(data='f30k-comparable-full',
            path_to_data='./data/',
            langs=['en', 'de'],
            margin=0.2,
            dim=100,
            dim_multimodal=100,
            dim_image=4096,
            dim_word=100,
            encoders={'en':'gru', 'de':'gru'}, # gru OR bow
            max_epochs=15,
            dispFreq=10,
            decay_c=0.,
            grad_clip=2.,
            maxlen_w=100,
            optimizer='adam',
            batch_size = 128,
            saveto='./models/f30k-comparable-full.npz',
            validFreq=100,
            testFreq=100,
            lrate=0.0002,
            reload_=False,
            # new parameters
            max_words={'en':0, 'de':0}, # integer, zero means unlimited
            debug=False,
            use_dropout=False,
            dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
            dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
            dropout_source=0.0, # dropout source words (0: no dropout)
            #dropout_prob=0.5,
            load_test=False,
            lambda_img_sent=0.5,
            lambda_sent_sent=0.5,
            bidirectional_enc=False,
            n_enc_hidden_layers=1,
            use_all_costs=False,
            create_dictionaries=False,
            attention_type='dot', # one of 'general', 'dot'
            decay_c_general_attention=0.0, # L2 regularisation for the attention matrices
            dictionaries_min_freq=0):

    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['langs'] = langs
    for lang in langs:
        model_options['encoder_%s'%lang] = encoders[lang]
        model_options['max_words_%s'%lang] = max_words[lang]
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_multimodal'] = dim_multimodal
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
    model_options['testFreq'] = testFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_
    model_options['use_dropout'] = use_dropout
    model_options['dropout_embedding'] = dropout_embedding
    model_options['dropout_hidden'] = dropout_hidden
    model_options['dropout_source'] = dropout_source
    #model_options['dropout_prob'] = dropout_prob
    model_options['bidirectional_enc'] = bidirectional_enc
    model_options['n_enc_hidden_layers'] = n_enc_hidden_layers
    model_options['load_test'] = load_test
    model_options['lambda_img_sent'] = lambda_img_sent
    model_options['lambda_sent_sent'] = lambda_sent_sent
    model_options['use_all_costs'] = use_all_costs
    model_options['use_all_costs'] = use_all_costs
    model_options['create_dictionaries'] = create_dictionaries
    model_options['dictionaries_min_freq'] = dictionaries_min_freq
    model_options['attention_type'] = attention_type
    model_options['decay_c_general_attention'] = decay_c_general_attention

    assert(n_enc_hidden_layers>=1)

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            models_options = pkl.load(f)

    # Load training and development sets, alternatively also test set
    print 'Loading dataset'
    train, dev, test = load_multilingual_dataset(data, langs,
            load_test=load_test)

    worddicts = []
    iworddicts = []
    if create_dictionaries:
        # Create and save dictionaries
        print 'Creating and saving multilingual dictionaries %s'%(", ".join(model_options['langs']))
        for lang_idx, lang in enumerate(langs):
            if load_test:
                worddict = build_dictionary(train[0][lang_idx]+dev[0][lang_idx]+test[0][lang_idx],
                                            dictionaries_min_freq)[0]
            else:
                worddict = build_dictionary(train[0][lang_idx]+dev[0][lang_idx],
                                            dictionaries_min_freq)[0]
            n_words_dict = len(worddict)
            print 'minimum word frequency: %i'%dictionaries_min_freq
            print '%s dictionary size: %s'%(lang,str(n_words_dict))
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
    else:
        # load dictionaries
        print 'Loading multilingual dictionaries %s'%(", ".join(model_options['langs']))
        for lang_idx, lang in enumerate(langs):
            with open('%s.dictionary-%s.pkl'%(saveto,lang), 'wb') as f:
                worddict = pkl.load(f)

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

    trng, inps, cost = build_model(tparams, model_options)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            # all parameters but general attention, if any
            if not kk.endswith('mapping'):
                weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # weight decay for the general attention, if applicable
    if decay_c_general_attention > 0. and attention_type == 'general':
        decay_g = theano.shared(numpy.float32(decay_c_general_attention), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk.endswith('mapping'):
                print 'Adding L2 for %s ...'%kk
                weight_decay += (vv ** 2).sum()
        weight_decay *= decay_g
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

    # create training set iterator where
    # a heuristic tries to make sure sentences in minibatch have a similar size
    train_iter = homogeneous_data_multilingual.HomogeneousDataMultilingual(train,
                batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    
    curr_best_model = None
    best_model_changed = True
    curr_best_score = 0.
    curr_best_rank  = 1e10
    curr_ranks_langs  = [1e10]*len(model_options['langs'])
    curr_scores_langs = [0.]*len(model_options['langs'])
    n_samples = 0

    ep_start = time.time()
    ep_times = [ ep_start ]
    for eidx in xrange(max_epochs):
        print 'Epoch ', eidx

        for xs, im in train_iter:
            uidx += 1
            xs, masks, im = homogeneous_data_multilingual.prepare_data(xs, im, \
                                                                       worddicts, \
                                                                       model_options=model_options, \
                                                                       maxlen=maxlen_w)

            if xs[0] is None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            # do not train on certain small sentences (less than 3 words)
            #if not x_src.shape[0]>=minlen_w and x_tgt.shape[0]>= minlen_w:
            #if not all( x.shape[0]>=minlen_w for x in xs ):
            #    print "At least one minibatch (in one of the languages in the model)",
            #    print "has less words than %i. Skipping..."%minlen_w
            #    skipped_samples += xs[0].shape[1]
            #    uidx -= 1
            #    continue

            n_samples += len(xs[0])

            # Update
            ud_start = time.time()
            # flatten inputs for theano function
            inps_ = []
            inps_.extend(xs)
            inps_.extend(masks)
            inps_.append(im)
            #cost = f_grad_shared(xs, masks, im)
            cost = f_grad_shared(*inps_)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing results...'
                curr_model = {}
                curr_model['options'] = model_options
                # store model's language dependent parameters
                for lang, worddict, iworddict, f_senc in zip(langs, worddicts, iworddicts, f_sencs):
                    curr_model['worddict_%s'%lang] = worddict
                    curr_model['wordidict_%s'%lang] = iworddict
                    curr_model['f_senc_%s'%lang] = f_senc
                curr_model['f_ienc'] = f_ienc

                # up-to-date model parameters
                params_ = unzip(tparams)

                # encode sentences
                lss = []
                for lang_idx, lang in enumerate(model_options['langs']):
                    ls  = encode_multilingual_sentences(curr_model, dev[0][lang_idx], lang=lang)
                    lss.append(ls)

                # encode images
                n_sentences_per_image = int(dev[2]) if len(dev)==3 else 1
                dev_img_feats = numpy.repeat(dev[1], n_sentences_per_image, axis=0).astype('float32')
                lim = encode_images(curr_model, dev_img_feats)

                # print scores for each language pair
                for lang_idx1, lang1 in enumerate(model_options['langs']):
                    for lang_idx2, lang2 in enumerate(model_options['langs']):
                        if lang_idx1 == lang_idx2 or lang_idx2 <= lang_idx1:
                            continue

                        sent_sent=None
                        # if attention type is general, pass on the mapping matrix
                        if attention_type == 'general':
                            sent_sent = params_['sentence_%i_sentence_%i_mapping'%(lang_idx1, lang_idx2)]

                        # text en to text de and vice-versa 
                        (r1, r5, r10, medr, medr_double) = \
                            t2t(lss[ lang_idx1 ], lss[ lang_idx2 ],
                                n_sentences_per_image=n_sentences_per_image,
                                attention_type=attention_type, sent_sent=sent_sent)
                        print "%s text to %s text: %.1f, %.1f, %.1f, %d (%.2f)" % (lang1, lang2, r1, r5, r10, medr, medr_double)

                        (r1, r5, r10, medr, medr_double) = \
                            t2t(lss[ lang_idx2 ], lss[ lang_idx1 ],
                                n_sentences_per_image=n_sentences_per_image,
                                attention_type=attention_type,
                                sent_sent=(sent_sent.T if sent_sent is not None else sent_sent))

                        print "%s text to %s text: %.1f, %.1f, %.1f, %d (%.2f)" % (lang2, lang1, r1, r5, r10, medr, medr_double)

                # compute scores
                currranks = 0. # the lower the better
                currscore = 0. # the higher the better
                for lang_idx1, lang1 in enumerate(model_options['langs']):
                    sent_img = None
                    # if attention type is general, pass on the mapping matrix
                    if attention_type == 'general':
                        sent_img = params_['image_sentence_%i_mapping'%lang_idx1]

                    (r1, r5, r10, medr, medr_double) = \
                            i2t(lim, lss[ lang_idx1 ],
                                n_sentences_per_image=n_sentences_per_image,
                                attention_type=attention_type, sent_img=sent_img)
                    print "Image to %s text: %.1f, %.1f, %.1f, %d (%.2f)" % (lang1, r1, r5, r10, medr, medr_double)

                    (r1i, r5i, r10i, medri, medr_doublei) = \
                            t2i(lim, lss[ lang_idx1 ],
                                n_sentences_per_image=n_sentences_per_image,
                                attention_type=attention_type, sent_img=sent_img)
                    print "%s text to image: %.1f, %.1f, %.1f, %d (%.2f)" % (lang1, r1i, r5i, r10i, medri, medr_doublei)

                    # adjust current overall score including all languages
                    currranks += medr_double + medr_doublei
                    currscore += r1 + r5 + r10 + r1i + r5i + r10i

                    # best current score for individual language/image pair
                    currranks_lang = medr_double + medr_doublei
                    currscore_lang = r1 + r5 + r10 + r1i + r5i + r10i

                    # first, we select the model that ranks best (median rank).
                    # second, if there is a tie, we select the model that has best sum of scores (recall@k).
                    if currranks_lang < curr_ranks_langs[ lang_idx1 ]:
                        curr_ranks_langs[ lang_idx1 ] = currranks_lang
                        curr_scores_langs[ lang_idx1 ] = currscore_lang

                        # save model
                        print 'saving best %s...' % lang1,
                        #params = unzip(tparams)
                        numpy.savez('%s.best-%s'%(saveto, lang1), **params_)
                        pkl.dump(model_options, open('%s.best-%s.pkl'%(saveto, lang1), 'wb'))
                        print 'done'

                    elif currranks_lang == curr_ranks_langs[ lang_idx1 ]:
                        print '%s ranks are equal the current best (=%i) ...'%( lang1, currranks_lang )
                        if currscore_lang > curr_scores_langs[ lang_idx1 ]:
                            curr_scores_langs[ lang_idx1 ] = currscore_lang

                            # save model
                            print 'saving best %s...' % lang1,
                            #params = unzip(tparams)
                            numpy.savez('%s.best-%s'%(saveto, lang1), **params_)
                            pkl.dump(model_options, open('%s.best-%s.pkl'%(saveto, lang1), 'wb'))
                            print 'done'

                if currranks < curr_best_rank:
                    curr_best_rank = currranks
                    curr_best_score = currscore
                    best_model_changed = True

                    # Save model
                    print 'Saving best overall model (%s)...' % str("-".join(model_options['langs'])),
                    params = unzip(tparams)
                    numpy.savez(saveto, **params)
                    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                    print 'Done'
                elif currranks == curr_best_rank:
                    print 'global ranks are equal to the current best (=%i)...'%int(currranks)
                    if currscore > curr_best_score:
                        # adjust current best overall score if needed
                        curr_best_score = currscore
                        best_model_changed = True

                        # Save model
                        print 'Saving best overall model (%s)...' % str("-".join(model_options['langs'])),
                        params = unzip(tparams)
                        numpy.savez(saveto, **params)
                        pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                        print 'Done'

            if numpy.mod(uidx, testFreq) == 0:

                if not best_model_changed:
                    print '.. Best model on valid set did not change from previous evaluation on test set...'
                    print ''
                else:
                    print '.. Computing results on test set...'

                    # update current best model (on the valid set)
                    best_model_dev = load_model(saveto, verbose=False)

                    # encode sentences
                    lss = []
                    for lang_idx, lang in enumerate(model_options['langs']):
                        ls  = encode_multilingual_sentences(best_model_dev, test[0][lang_idx], lang=lang)
                        lss.append(ls)
                    #ls = encode_multilingual_sentences(best_model_dev, test[0])

                    n_sentences_per_image = test[2] if len(test)==3 else 1
                    test_img_feats = numpy.repeat(test[1], n_sentences_per_image, axis=0).astype('float32')
                    lim = encode_images(best_model_dev, test_img_feats)

                    for lang_idx1, lang1 in enumerate(model_options['langs']):
                        for lang_idx2, lang2 in enumerate(model_options['langs']):
                            if lang_idx1 == lang_idx2 or lang_idx2 <= lang_idx1:
                                continue

                            sent_sent=None
                            # if attention type is general, pass on the mapping matrix
                            if attention_type == 'general':
                                sent_sent = best_model_dev['tparams']['sentence_%i_sentence_%i_mapping'%(lang_idx1, lang_idx2)]

                            (r1, r5, r10, medr, medr_double) = \
                                    t2t(lss[ lang_idx1 ], lss[ lang_idx2 ],
                                        n_sentences_per_image=n_sentences_per_image,
                                        attention_type=attention_type, sent_sent=sent_sent)
                            print ".. %s text to %s text: %.1f, %.1f, %.1f, %d (%.2f)" % (lang1, lang2, r1, r5, r10, medr, medr_double)

                            (r1, r5, r10, medr, medr_double) = \
                                    t2t(lss[ lang_idx2 ], lss[ lang_idx1 ],
                                        n_sentences_per_image=n_sentences_per_image,
                                        attention_type=attention_type,
                                        sent_sent=(sent_sent.T if sent_sent is not None else sent_sent))
                            print ".. %s text to %s text: %.1f, %.1f, %.1f, %d (%.2f)" % (lang2, lang1, r1, r5, r10, medr, medr_double)

                    #for i in range(len(lss)):
                    for lang_idx1, lang1 in enumerate(model_options['langs']):
                        sent_img = None
                        # if attention type is general, pass on the mapping matrix
                        if attention_type == 'general':
                            sent_img = best_model_dev['tparams']['image_sentence_%i_mapping'%lang_idx1]

                        (r1, r5, r10, medr, medr_double) = i2t(lim, lss[ lang_idx1 ],
                                                  n_sentences_per_image=n_sentences_per_image,
                                                  attention_type=attention_type, sent_img=sent_img)
                        print ".. Image to %s text: %.1f, %.1f, %.1f, %d (%.2f)" % (lang1, r1, r5, r10, medr, medr_double)

                        (r1i, r5i, r10i, medri, medr_doublei) = t2i(lim, lss[ lang_idx1 ],
                                                      n_sentences_per_image=n_sentences_per_image,
                                                      attention_type=attention_type, sent_img=sent_img)
                        print ".. %s text to image: %.1f, %.1f, %.1f, %d (%.2f)" % (lang1, r1i, r5i, r10i, medri, medr_doublei)

                    best_model_changed = False
                    print ''

        ep_end = time.time()
        ep_times.append(ep_end)
        #print 'Seen %d samples'%n_samples

        seconds = ep_times[-1] - ep_times[0]
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print "Seen %i epoch(s) (%i samples) in %d:%02d:%02d" % (eidx, n_samples, h, m, s)

    seconds = ep_times[-1] - ep_times[0]
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print "Finished execution in %d:%02d:%02d" % (h, m, s)

if __name__=="__main__":
    print "Starting..."
    trainer()
