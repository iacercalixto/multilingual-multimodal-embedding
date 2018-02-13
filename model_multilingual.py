"""
Model specification
"""
import pdb
import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal
import numpy

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, l2norm, concatenate
from layers import shared_dropout_layer, get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # if using bidirectional RNN,
    # forward and backward embeddings are half the final MM embedding size because
    # they will be concatenated to form the sentence embedding
    #sent_dim = int(options['dim'])//2 if options['bidirectional_enc'] else int(options['dim'])
    sent_dim = options['dim']
    ctx_dim  = options['dim'] # context vector in case of mono RNN encoder

    langs = options['langs']
    for idx, lang in enumerate(langs):
        # word embeddings
        params['Wemb_%s'%lang] = norm_weight(options['n_words_%s'%lang], options['dim_word'])

        # encoder type (currently 'bow', 'gru' or 'lstm')
        if options['encoder_%s'%lang] != 'bow':
            for i in range(int(options['n_enc_hidden_layers'])):
                layer_name_prefix='encoder_%s_%i'%(lang,i)
                # first hidden layer has input word embeddings, next layers have input (hidden) sentence embeddings
                nin=options['dim_word'] if i==0 else sent_dim
                params = get_layer(options['encoder_%s'%lang])[0](options, params, prefix=layer_name_prefix,
                                                                  nin=nin, dim=sent_dim)
            if options['bidirectional_enc']:
                ctx_dim  = 2*options['dim'] # context vector in case of biRNN encoder

                for i in range(int(options['n_enc_hidden_layers'])):
                    layer_name_prefix='encoder_%s_r_%i'%(lang,i)
                    # first hidden layer has input word embeddings, next layers have input (hidden) sentence embeddings
                    nin=options['dim_word'] if i==0 else sent_dim
                    params = get_layer(options['encoder_%s'%lang])[0](options, params, prefix=layer_name_prefix,
                                                                      nin=nin, dim=sent_dim)

        # if using general attention, create matrices for each possible
        # sentence-image and sentence-sentence pairs
        if 'attention_type' in options and options['attention_type'] == 'general':
            # sentence_lang-image mapping
            params['image_sentence_%i_mapping'%idx] = \
                norm_weight(options['dim_multimodal'], options['dim_multimodal'], ortho=False)

            # sentence_lang1 - sentence_langN mappings
            for idx1, lang1 in enumerate(langs):
                if idx==idx1 or idx1<=idx: continue
                params['sentence_%i_sentence_%i_mapping'%(idx,idx1)] = \
                    norm_weight(options['dim_multimodal'], options['dim_multimodal'], ortho=False)

    # Sentence-multimodal projection
    params = get_layer('ff')[0](options, params, prefix='ff_sentence_mm', nin=ctx_dim, nout=options['dim_multimodal'])

    # Image-multimodal projection
    params = get_layer('ff')[0](options, params, prefix='ff_image_mm', nin=options['dim_image'], nout=options['dim_multimodal'])

    return params

def contrastive_loss(tparams, options, im, sents):
    """
    Compute contrastive loss.
    Contrastive loss is computed between each language-image pair
    but not beween sentences in different languages.
    """
    margin = options['margin']
    attention_type = options['attention_type'] if 'attention_type' in options else 'dot'
    n_langs = len(sents)

    final_cost = 0.
    # compute cost for each language and aggregate on final cost
    for i in range(n_langs):
        s_lang = sents[i]
        # compute image-sentence score matrix
        if attention_type == 'dot':
            scores_lang = tensor.dot(im, s_lang.T)

        elif attention_type == 'general':
            if attention_type == 'general':
                sents_img = tparams['image_sentence_%i_mapping'%i]
            scores_lang = im.dot( sents_img ).dot( s_lang.T )

        else:
            raise Exception("Attention type not supported: %s"%attention_type)

        diagonal_lang = scores_lang.diagonal()
        # cost over sentence
        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_sent_lang = tensor.maximum(0, margin - diagonal_lang + scores_lang)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im_lang = tensor.maximum(0, margin - diagonal_lang.reshape((-1, 1)) + scores_lang)
        # clear diagonals
        cost_sent_lang = fill_diagonal(cost_sent_lang, 0)
        cost_im_lang = fill_diagonal(cost_im_lang, 0)

        # aggregate
        final_cost += cost_sent_lang.sum() + cost_im_lang.sum()

    return final_cost

def contrastive_loss_all(tparams, options, im, sents, lambda_img_sent=0.5, lambda_sent_sent=0.5):
    """
    Compute contrastive loss.
    Contrastive loss is computed between each language-image pair
    as well as beween sentences in different languages.
    """
    margin = options['margin']
    attention_type = options['attention_type'] if 'attention_type' in options else 'dot'

    n_langs = len(sents)

    final_cost = 0.
    # compute costs for each language-image pair and aggregate final cost
    for i in range(n_langs):
        # compute image-sentence subcost
        s_lang = sents[i]

        # compute image-sentence score matrix
        #scores_lang = tensor.dot(im, s_lang.T)
        if attention_type == 'dot':
            scores_lang = tensor.dot(im, s_lang.T)

        elif attention_type == 'general':
            if attention_type == 'general':
                # get matrix to map sentences and images
                sents_img = tparams['image_sentence_%i_mapping'%i]
            scores_lang = im.dot( sents_img ).dot( s_lang.T )

        else:
            raise Exception("Attention type not supported: %s"%attention_type)

        diagonal_lang = scores_lang.diagonal()
        # cost over sentence
        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_sent_lang = tensor.maximum(0, margin - diagonal_lang + scores_lang)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im_lang = tensor.maximum(0, margin - diagonal_lang.reshape((-1, 1)) + scores_lang)
        # clear diagonals
        cost_sent_lang = fill_diagonal(cost_sent_lang, 0)
        cost_im_lang = fill_diagonal(cost_im_lang, 0)

        # aggregate
        final_cost += lambda_img_sent * (cost_sent_lang.sum() + cost_im_lang.sum())

    # compute costs for each-language-language pair and aggregate final cost
    for i in range(n_langs):
        for j in range(n_langs):
            if i==j or j<=i:
                continue
            # compute sentence-sentence subcost
            s_lang1 = sents[i]
            s_lang2 = sents[j]

            # compute sent1-sent2 score matrix
            #scores_lang = tensor.dot(s_lang1, s_lang2.T)
            if attention_type == 'dot':
                scores_lang = tensor.dot(s_lang1, s_lang2.T)

            elif attention_type == 'general':
                # get matrices to map sentences in different languages
                sents_sents = tparams['sentence_%i_sentence_%i_mapping'%(i,j)]
                scores_lang = s_lang1.dot( sents_sents ).dot( s_lang2.T )

            else:
                raise Exception("Attention type not supported: %s"%attention_type)

            diagonal_lang = scores_lang.diagonal()
            # cost over sent1
            # compare every diagonal score to scores in its column (i.e, all contrastive sent2 for each sent1)
            cost_sent1 = tensor.maximum(0, margin - diagonal_lang + scores_lang)
            # compare every diagonal score to scores in its row (i.e, all contrastive sent1 for each sent2)
            cost_sent2 = tensor.maximum(0, margin - diagonal_lang.reshape((-1, 1)) + scores_lang)
            # clear diagonals
            cost_sent1 = fill_diagonal(cost_sent1, 0)
            cost_sent2 = fill_diagonal(cost_sent2, 0)

            # aggregate
            final_cost += lambda_sent_sent * (cost_sent1.sum() + cost_sent2.sum())

    return final_cost

def build_model(tparams, options):                                                                                           
    """
    Computation graph for the model
    """
    opt_ret = dict()
    use_noise = theano.shared(numpy.asarray(1., dtype=theano.config.floatX))
    try:
        trng = RandomStreams(1234, use_cuda=True)
    except:
        print "Could not apply use_cuda==True in RandonStreams ..."
        trng = RandomStreams(1234)

    xs = []
    xmasks = []

    langs = options['langs']
    for lang in langs:
        # description string: #words x #samples
        x_lang = tensor.matrix('x_%s'%lang, dtype='int64')
        mask_lang = tensor.matrix('mask_%s'%lang, dtype='float32')
        xs.append(x_lang)
        xmasks.append(mask_lang)

    xs_r = []
    xmasks_r = []
    if options['bidirectional_enc']:
        for i,lang in enumerate(langs):
            x_lang = xs[i]
            mask_lang = xmasks[i]
            # reverse
            x_lang_r = x_lang[::-1]
            mask_lang_r = mask_lang[::-1]

            xs_r.append(x_lang_r)
            xmasks_r.append(mask_lang_r)

    sents_all = []
    im = tensor.matrix('im', dtype='float32')
    n_samples = im.shape[0]

    for i,lang in enumerate(langs):
        x_lang = xs[i]
        mask_lang = xmasks[i]

        n_timesteps_lang = x_lang.shape[0]
        n_samples_lang = x_lang.shape[1]

        if options['use_dropout']:
            # dropout probs for the word embeddings
            retain_probability_emb = 1-options['dropout_embedding']
            # dropout probs for the RNN hidden states
            retain_probability_hidden = 1-options['dropout_hidden']
            # dropout probs for the source words
            retain_probability_source = 1-options['dropout_source']
            # hidden states
            rec_dropout = shared_dropout_layer((2, n_samples_lang, options['dim']), use_noise, trng, retain_probability_hidden)
            rec_dropout_r = shared_dropout_layer((2, n_samples_lang, options['dim']), use_noise, trng, retain_probability_hidden)
            # word embeddings
            emb_dropout = shared_dropout_layer((2, n_samples_lang, options['dim_word']), use_noise, trng, retain_probability_emb)
            emb_dropout_r = shared_dropout_layer((2, n_samples_lang, options['dim_word']), use_noise, trng, retain_probability_emb)
            # source words
            source_dropout = shared_dropout_layer((n_timesteps_lang, n_samples_lang, 1), use_noise, trng, retain_probability_source)
            source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))
        else:
            # hidden states
            rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
            rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
            # word embeddings
            emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
            emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))

        # Word embedding (for a particular language `lang`)
        # forward
        emb_lang = tparams['Wemb_%s'%lang][x_lang.flatten()]
        emb_lang = emb_lang.reshape([n_timesteps_lang, n_samples_lang, options['dim_word']])

        if options['use_dropout']:
            emb_lang *= source_dropout

        if options['bidirectional_enc']:
            x_lang_r = xs_r[i]
            mask_lang_r = xmasks_r[i]

            # backward lang encoder
            emb_lang_r = tparams['Wemb_%s'%lang][x_lang_r.flatten()]
            emb_lang_r = emb_lang_r.reshape([n_timesteps_lang, n_samples_lang, options['dim_word']])

            if options['use_dropout']:
                emb_lang_r *= source_dropout[::-1]

        # Encode sentence in language `lang`
        if options['encoder_%s'%lang] == 'bow':
            sents_lang = (emb_lang * mask_lang[:,:,None]).sum(0)
        else:
            # iteratively push input from first hidden layer until the last
            for i in range(int(options['n_enc_hidden_layers'])):
                layer_name_prefix='encoder_%s_%i'%(lang,i)
                # if first hidden layer use wembs, otherwise output of previous hidden layer
                layer_below=emb_lang if i==0 else layer_below[0]

                # do not apply dropout on word embeddings layer
                #if options['use_dropout'] and i>0:
                #    layer_below = dropout_layer(layer_below, use_noise, trng, prob=options['dropout_prob'])

                layer_below=get_layer(options['encoder_%s'%lang])[1](tparams,
                        layer_below, options, None, prefix=layer_name_prefix, mask=mask_lang,
                        emb_dropout=emb_dropout, 
                        rec_dropout=rec_dropout)

                if i==int(options['n_enc_hidden_layers'])-1:
                    # sentence embeddings (projections) are the output of the last hidden layer
                    proj_lang = layer_below

            # apply forward and backward steps and concatenate both
            if options['bidirectional_enc']:
                # concatenate forward and backward pass RNNs
                # iteratively push input from first hidden layer until the last
                for i in range(int(options['n_enc_hidden_layers'])):
                    layer_name_prefix='encoder_%s_r_%i'%(lang,i)
                    # if first hidden layer use wembs, else output of prev hidden layer
                    layer_below=emb_lang_r if i==0 else layer_below[0]

                    # do not apply dropout on word embeddings layer
                    #if options['use_dropout'] and i>0:
                    #    layer_below = dropout_layer(layer_below, use_noise, trng, prob=options['dropout_prob'])

                    layer_below=get_layer(options['encoder_%s'%lang])[1](tparams,
                            layer_below, options, None,
                            prefix=layer_name_prefix,
                            mask=mask_lang_r,
                            emb_dropout=emb_dropout_r,
                            rec_dropout=rec_dropout_r)

                    if i==int(options['n_enc_hidden_layers'])-1:
                        # sentence embeddings (projections) are the output of the last hidden layer
                        proj_lang_r = layer_below

                # use the last state of forward + backward encoder rnns
                sents_lang = concatenate([proj_lang[0][-1], proj_lang_r[0][-1]], axis=proj_lang[0].ndim-2)
            else:
                sents_lang = proj_lang[0][-1]

        if options['use_dropout']:
            sents_lang *= shared_dropout_layer((n_samples_lang, options['dim']), use_noise, trng, retain_probability_hidden)

        # project sentences into multimodal space
        sents_mm = get_layer('ff')[1](tparams, sents_lang, options, prefix='ff_sentence_mm', activ='linear')

        if options['attention_type'] == 'dot':
            sents_mm = l2norm(sents_mm)

        if options['use_dropout']:
            sents_mm *= shared_dropout_layer((n_samples_lang, options['dim_multimodal']), use_noise, trng, retain_probability_hidden)

        sents_all.append(sents_mm)

    # Encode images
    images = get_layer('ff')[1](tparams, im, options, prefix='ff_image_mm', activ='linear')

    if options['attention_type'] == 'dot':
        images = l2norm(images)

    if options['use_dropout']:
        images *= shared_dropout_layer((n_samples, options['dim_multimodal']), use_noise, trng, retain_probability_hidden)

    # Compute loss
    lambda_img_sent  = options['lambda_img_sent']
    lambda_sent_sent = options['lambda_sent_sent']
    if options['use_all_costs']:
        cost = contrastive_loss_all(tparams, options,
                                    images, sents_all, lambda_img_sent, lambda_sent_sent)
    else:
        cost = contrastive_loss(tparams, options,
                                images, sents_all)

    # return flattened inputs
    inps = []
    inps.extend(xs)
    inps.extend(xmasks)
    inps.append(im)

    return trng, inps, cost


def build_sentence_encoders(tparams, options):
    """
    Sentence encoder only to be used at test time
    """
    opt_ret = dict()
    trng = RandomStreams(1234)

    #xs, masks, sents_all = [], [], []
    in_outs = []

    langs = options['langs']
    for lang in langs:
        # description string: #words x #samples
        # forward
        x = tensor.matrix('x_%s'%lang, dtype='int64')
        mask = tensor.matrix('x_mask_%s'%lang, dtype='float32')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # Word embedding (forward)
        emb = tparams['Wemb_%s'%lang][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

        if options['bidirectional_enc']:
            # backward RNN
            x_r = x[::-1]
            mask_r = mask[::-1]
            emb_r = tparams['Wemb_%s'%lang][x_r.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

        if options['use_dropout']:
            retain_probability_emb = 1-options['dropout_embedding']
            retain_probability_hidden = 1-options['dropout_hidden']
            retain_probability_source = 1-options['dropout_source']
            rec_dropout = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
            rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
            emb_dropout = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
            emb_dropout_r = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
            source_dropout = theano.shared(numpy.float32(retain_probability_source))
            emb *= source_dropout
            if options['bidirectional_enc']:
                embr *= source_dropout
        else:
            rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
            rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
            emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
            emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))

        # Encode sentences
        if options['encoder_%s'%lang] == 'bow':
            sents = (emb * mask[:,:,None]).sum(0)
        else:
            # iteratively push input from first hidden layer until the last
            for i in range(int(options['n_enc_hidden_layers'])):
                layer_name_prefix='encoder_%s_%i'%(lang,i)
                # if first layer input are wembs, otherwise input will be output of last hidden layer
                layer_below=emb if i==0 else layer_below[0]
                layer_below=get_layer(options['encoder_%s'%lang])[1](tparams,
                        layer_below, options, None, prefix=layer_name_prefix, mask=mask,
                        emb_dropout=emb_dropout, rec_dropout=rec_dropout)

                if i==int(options['n_enc_hidden_layers'])-1:
                    # sentence embeddings (projections) are the output of the last hidden layer
                    proj = layer_below

            if options['bidirectional_enc']:
                for i in range(int(options['n_enc_hidden_layers'])):
                    layer_name_prefix='encoder_%s_r_%i'%(lang,i)
                    # if first layer input are wembs, otherwise input will be output of last hidden layer
                    layer_below=emb_r if i==0 else layer_below[0]
                    layer_below=get_layer(options['encoder_%s'%lang])[1](tparams,
                            layer_below, options, None, prefix=layer_name_prefix, mask=mask_r,
                            emb_dropout=emb_dropout_r, rec_dropout=rec_dropout_r)

                    if i==int(options['n_enc_hidden_layers'])-1:
                        # sentence embeddings (projections) are the output of the last hidden layer
                        proj_r = layer_below

                # use last hidden state of forward and backward RNNs
                sents = concatenate([proj[0][-1],proj_r[0][-1]], axis=proj[0].ndim-2)
            else:
                sents = proj[0][-1]

        if options['use_dropout']:
            sents *= shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden)

        # project sentences into multimodal space
        sents_mm = get_layer('ff')[1](tparams, sents, options, prefix='ff_sentence_mm', activ='linear')
        if not 'attention_type' in options or options['attention_type'] == 'dot':
            sents_mm = l2norm(sents_mm)

        if options['use_dropout']:
            sents_mm *= shared_dropout_layer((n_samples, options['dim_multimodal']), use_noise, trng, retain_probability_hidden)

        # outputs per language
        in_outs.append(([x, mask], sents_mm))

    return trng, in_outs


def build_image_encoder(tparams, options):
    """
    Encoder only, for images
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # image features
    im = tensor.matrix('im', dtype='float32')

    # Encode images
    images_mm = get_layer('ff')[1](tparams, im, options, prefix='ff_image_mm', activ='linear')
    if not 'attention_type' in options or options['attention_type'] == 'dot':
        images_mm = l2norm(images_mm)
    
    if options['use_dropout']:
        images_mm *= shared_dropout_layer((n_samples, options['dim_multimodal']), use_noise, trng, retain_probability_hidden)

    return trng, [im], images_mm

