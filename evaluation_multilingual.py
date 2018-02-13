"""
Evaluation code for multilingual, multimodal ranking
"""
import numpy

from datasets_multilingual import load_multilingual_dataset
from tools_multilingual import encode_multilingual_sentences, encode_images

def evalrank(model, data, split='dev'):
    """
    Evaluate a trained model on either dev or test
    """
    attention_type = model['options']['attention_type']

    print 'Loading dataset'
    if split == 'dev':
        X = load_multilingual_dataset(data, langs=model['options']['langs'], load_train=False, load_test=False)[1]
    else:
        X = load_multilingual_dataset(data, langs=model['options']['langs'], load_train=False, load_dev=False)[2]

    n_sentences_per_image = X[0][2]

    print 'Computing results...'
    lss = []
    for lang_idx, language in enumerate(model['options']['langs']):
        print 'Language: %s'%language
        print X[0][lang_idx][0], X[0][lang_idx][1]

        ls = encode_multilingual_sentences(model, X[0][lang_idx], lang=language)
        lim = encode_images(model, X[1])

        # if attention type is general, pass on the mapping matrix
        if attention_type == 'general':
            sent_img = model['tparams']['image_sentence_%i_mapping'%lang_idx]

        (r1, r5, r10, medr, medr_double) = \
                i2t(lim, ls,
                    n_sentences_per_image=n_sentences_per_image,
                    attention_type=attention_type, sent_img=sent_img)
        print "Image to %s text: %.1f, %.1f, %.1f, %d (%.2f)" % (language, r1, r5, r10, medr, medr_double)

        (r1i, r5i, r10i, medri, medr_doublei) = \
                t2i(lim, ls,
                    n_sentences_per_image=n_sentences_per_image,
                    attention_type=attention_type, sent_img=sent_img)
        print "%s text to image: %.1f, %.1f, %.1f, %d (%.2f)" % (language, r1i, r5i, r10i, medri, medr_doublei)

        lss.append(ls)

    # if attention type is general, pass on the mapping matrix
    if attention_type == 'general':
        for lang_idx1, _ in enumerate(model['options']['langs']):
            for lang_idx2, _ in enumerate(model['options']['langs']):
                if lang_idx1 == lang_idx2 or lang_idx2 <= lang_idx1:
                    continue

                #lang_idx1, lang_idx2 = 0, 1
                sent_sent = model['tparams']['sentence_%i_sentence_%i_mapping'%(lang_idx1, lang_idx2)]

                print 'Computing En <-> De results...'
                (r1, r5, r10, medr, medr_double) = t2t(lss[0], lss[1], n_sentences_per_image=n_sentences_per_image,
                                          attention_type=attention_type, sent_sent=sent_sent)
                print "en to de: %.1f, %.1f, %.1f, %d (%.2f)" % (r1, r5, r10, medr, medr_double)

                (r1, r5, r10, medr, medr_double) = t2t(lss[1], lss[0], n_sentences_per_image=n_sentences_per_image,
                                          attention_type=attention_type, sent_sent=sent_sent)
                print "de to en: %.1f, %.1f, %.1f, %d (%.2f)" % (r1, r5, r10, medr, medr_double)
    else:
        print 'Computing En <-> De results...'
        (r1, r5, r10, medr, medr_double) = t2t(lss[0], lss[1], n_sentences_per_image=n_sentences_per_image,
                                  attention_type=None, sent_sent=None)
        print "en to de: %.1f, %.1f, %.1f, %d (%.2f)" % (r1, r5, r10, medr, medr_double)

        (r1, r5, r10, medr, medr_double) = t2t(lss[1], lss[0], n_sentences_per_image=n_sentences_per_image,
                                  attention_type=None, sent_sent=None)
        print "de to en: %.1f, %.1f, %.1f, %d (%.2f)" % (r1, r5, r10, medr, medr_double)

def i2t(images, captions, npts=None, n_sentences_per_image=5, attention_type=None, sent_img=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    #sent_img = model['tparams']['sentence_%i_image_mapping'%lang_idx]
    #attention_type = model['options']['attention_type']

    if npts == None:
        #npts = images.shape[0] / 5
        npts = images.shape[0] / n_sentences_per_image
    index_list = []

    #print "images.shape: %s"%str(images.shape)
    #print "captions.shape: %s"%str(captions.shape)

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        #im = images[5 * index].reshape(1, images.shape[1])
        im = images[n_sentences_per_image * index].reshape(1, images.shape[1])

        # Compute scores
        if attention_type == 'dot':
            d = numpy.dot(im, captions.T).flatten()
        elif attention_type == 'general':
            d = im.dot( sent_img ).dot( captions.T ).flatten()
        else:
            raise Exception("Attention type not implemented: %s"%attention_type)
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        #for i in range(5*index, 5*index + 5, 1):
        for i in range(n_sentences_per_image*index, n_sentences_per_image*index + n_sentences_per_image, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    mean = numpy.mean(ranks)
    return (r1, r5, r10, medr, mean)

def t2i(images, captions, npts=None, n_sentences_per_image=5, attention_type=None, sent_img=None):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        #npts = images.shape[0] / 5
        npts = images.shape[0] / n_sentences_per_image
    #ims = numpy.array([images[i] for i in range(0, len(images), 5)])
    ims = numpy.array([images[i] for i in range(0, len(images), n_sentences_per_image)])

    #print "ims.shape: %s"%str(ims.shape)
    #print "captions.shape: %s"%str(captions.shape)

    #ranks = numpy.zeros(5 * npts)
    ranks = numpy.zeros(n_sentences_per_image * npts)
    for index in range(npts):

        # Get query captions
        #queries = captions[5*index : 5*index + 5]
        queries = captions[n_sentences_per_image*index : n_sentences_per_image*index + n_sentences_per_image]

        # Compute scores
        #d = numpy.dot(queries, ims.T)
        if attention_type == 'dot':
            d = numpy.dot(queries, ims.T)
        elif attention_type == 'general':
            d = queries.dot( sent_img.T ).dot( ims.T )
        else:
            raise Exception("Attention type not implemented: %s"%attention_type)

        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            #ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            ranks[n_sentences_per_image * index + i] = numpy.where(inds[i] == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    mean = numpy.mean(ranks)
    return (r1, r5, r10, medr, mean)

def t2t(captions_src, captions_tgt, npts=None, n_sentences_per_image=5, attention_type=None, sent_sent=None):
    """
    Text src->Text tgt (Multilingual search)
    Captions src: (5N, K) matrix of captions in the source language
    Captions tgt: (5N, K) matrix of captions in the target language
    """
    if npts == None:
        #npts = captions_src.shape[0] / 5
        npts = captions_src.shape[0] / n_sentences_per_image
        #npts = images.shape[0] / 5
    index_list = []

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query sentences
        sents_src = captions_src[n_sentences_per_image * index].reshape(1, captions_src.shape[1])
        #print "sents_src.shape: %s"%str(sents_src.shape)
        #print "captions_tgt.shape: %s"%str(captions_tgt.shape)

        # Compute scores - i2t
        if attention_type == 'dot':
            d = numpy.dot(sents_src, captions_tgt.T).flatten()
        elif attention_type == 'general':
            d = sents_src.dot( sent_sent ).dot( captions_tgt.T ).flatten()
        else:
            raise Exception("Attention type not implemented: %s"%attention_type)

        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(n_sentences_per_image*index, n_sentences_per_image*index + n_sentences_per_image, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    mean = numpy.mean(ranks)
    return (r1, r5, r10, medr, mean)

