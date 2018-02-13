import os, sys
import argparse
import numpy
from tools_multilingual import load_model, encode_multilingual_sentences, encode_images
from datasets_multilingual import load_multilingual_dataset
from evaluation_multilingual import i2t, t2i, t2t, evalrank

if __name__=="__main__":
    # prevent stdout buffering
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    parser = argparse.ArgumentParser(description="Evaluate multilingual multimodal embedding.")
    parser.add_argument('--data', required=True, type=str, help="Full path to data set.")
    parser.add_argument('--model-options', default=None, type=str, help='Model options file pickled (optional).')
    parser.add_argument('--dictionaries', default=None, type=str, nargs="+", help='Dictionaries file pickled (optional).')
    parser.add_argument('--saveto', required=True, type=str, help='Full path to model file.')
    parser.add_argument('--split', default='test', type=str, choices=['test', 'dev'], help="Split to evaluate.")
    parser.add_argument('--langs', default=['en', 'de'], type=str, nargs="+", help='Languages of the model')
    args = parser.parse_args()

    saveto = args.saveto
    split = args.split
    data = args.data
    langs = args.langs
    options = args.model_options if not args.model_options is None else "%s.pkl"%saveto
    dictionaries = []
    if args.dictionaries is None:
        for lang in langs:
            dictionaries.append("%s.dictionary-%s.pkl"%(saveto,lang))
    else:
        dictionaries = args.dictionaries

    load_test = True if split=='test' else False
    # Load split
    _, dev, test = load_multilingual_dataset(data, langs=langs,
        load_train=False, load_dev=not load_test, load_test=load_test)

    # Validate against the desired split
    test = test if split=="test" else dev

    print 'Computing results on %s set...'%split
    # load current best model on the valid set
    best_model_dev = load_model(saveto, options, dictionaries, verbose=False)

    # encode sentences
    lss = []
    for lang_idx, lang in enumerate(langs):
        ls  = encode_multilingual_sentences(best_model_dev, test[0][lang_idx], lang=lang)
        lss.append(ls)
    #ls = encode_multilingual_sentences(best_model_dev, test[0])

    n_sentences_per_image = test[2] if len(test)==3 else 1
    test_img_feats = numpy.repeat(test[1], n_sentences_per_image, axis=0).astype('float32')
    lim = encode_images(best_model_dev, test_img_feats)

    print 'n_sentences_per_image: %i'%int(n_sentences_per_image)

    attention_type = best_model_dev['options']['attention_type']
    #for lang_idx in range(2):
    for lang_idx, lang in enumerate(langs):
        # if attention type is general, pass on the mapping matrix
        sent_img=None
        if attention_type == 'general':
            sent_img = best_model_dev['tparams']['image_sentence_%i_mapping'%lang_idx]

        # compute and print image-to-sentence rankings
        (r1, r5, r10, medr, mean) = i2t(lim, lss[ lang_idx ], n_sentences_per_image=n_sentences_per_image,
                                  attention_type=attention_type, sent_img=sent_img)
        print "Image to text (%s): %.1f, %.1f, %.1f, %d (%.2f)" % (lang, r1, r5, r10, medr, mean)

        (r1i, r5i, r10i, medri, meani) = t2i(lim, lss[ lang_idx ], n_sentences_per_image=n_sentences_per_image,
                                  attention_type=attention_type, sent_img=sent_img)
        print "Text (%s) to image: %.1f, %.1f, %.1f, %d (%.2f)" % (lang, r1i, r5i, r10i, medri, meani)

    for lang_idx1, lang1 in enumerate(langs):
        for lang_idx2, lang2 in enumerate(langs):
            if lang_idx1 == lang_idx2 or lang_idx2 <= lang_idx1:
                continue

            sent_sent=None
            if attention_type == 'general':
                sent_sent = best_model_dev['tparams']['sentence_%i_sentence_%i_mapping'%(lang_idx1, lang_idx2)]

            # compute and print sentence-to-sentence rankings
            (r1, r5, r10, medr, mean) = t2t(lss[ lang_idx1 ], lss[ lang_idx2 ], n_sentences_per_image=n_sentences_per_image,
                                      attention_type=attention_type,
                                      sent_sent=sent_sent)
            print "Text (%s) to text (%s): %.1f, %.1f, %.1f, %d (%.2f)" % (lang1, lang2, r1, r5, r10, medr, mean)

            (r1, r5, r10, medr, mean) = t2t(lss[ lang_idx2 ], lss[ lang_idx1 ], n_sentences_per_image=n_sentences_per_image,
                                      attention_type=attention_type,
                                      sent_sent=(sent_sent.T if sent_sent is not None else sent_sent))
            print "Text (%s) to text (%s): %.1f, %.1f, %.1f, %d (%.2f)" % (lang2, lang1, r1, r5, r10, medr, mean)

