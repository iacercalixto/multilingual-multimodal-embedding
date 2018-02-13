import argparse
import sys, os
from collections import OrderedDict
from train_multilingual import trainer


if __name__ == '__main__':
    # prevent stdout buffering
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    parser = argparse.ArgumentParser(description="Train multilingual multimodal embedding.")

    parser.add_argument('--data', default=None, type=str,
                        help="Model name. (currently for bilingual embedding must be 'wikiMM').")
    parser.add_argument('--saveto', default=None, type=str, required=True,
                        help='Full path to model file.')
    parser.add_argument('--minimum-sentence-length', default=1, type=int,
                        help='Minimum sentence length in either source or target languages.')
    parser.add_argument('--maximum-sentence-length', default=100, type=int,
                        help='Maximum sentence length in either source or target languages.')
    parser.add_argument('--word-embeddings-dimensionality', default=620, type=int,
                        help='Dimensionality of word embedding.')
    parser.add_argument('--hidden-layer-dimensionality', default=1048, type=int,
                        help='Dimensionality of the hidden layer of the RNN (if bidir, each RNN has this dimension).')
    parser.add_argument('--multimodal-embeddings-dimensionality', default=2048, type=int,
                        help='Dimensionality of multimodal embedding.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Margin parameter to be used in the pairwise ranking loss optimisation.')
    parser.add_argument('--max-epochs', default=20, type=int,
                        help='Maximum number of epochs to train (i.e., number of times '+\
                        'the training set is iterated over by the model).')
    parser.add_argument('--use-dropout', action='store_true', help='Use dropout (0.5).')
    #parser.add_argument('--dropout-prob', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--dropout-embedding', type=float, default=0.5, help='Dropout probability for word embeddings.')
    parser.add_argument('--dropout-hidden', type=float, default=0.5, help='Dropout probability for hidden layers.')
    parser.add_argument('--dropout-source', type=float, default=0.5, help='Dropout probability for source word tokens.')
    parser.add_argument('--decay-c', type=float, default=0.0, help='L2 regularisation weight.')
    parser.add_argument('--bidirectional-encoder', action='store_true',
                        help='Use bidirectional RNNs for encoding sentences.')
    parser.add_argument('--n-encoder-hidden-layers', type=int, default=1,
                        help='Number of hidden layers to use in RNN encoders.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Mini-batch size.')

    parser.add_argument('--debug', default=False, action='store_true', help='Debug mode.')
    parser.add_argument('--reload', default=False, action='store_true', help='Reload model previously trained.')
    parser.add_argument('--create-dictionaries', default=False, action='store_true', help='Create dictionaries from scratch.')
    parser.add_argument('--dictionaries-min-freq', default=0, type=int,
                        help='Minimum word frequency to be included in the dictionary.')
    parser.add_argument('--load-test', default=False, action='store_true', help='Load test set.')

    # parameters exclusive to multilingual embeddings
    parser.add_argument('--langs', default=None, type=str, nargs="+",
                        help='Languages to be used for encoders in multilingual model '+\
                        '(eg. \'en\', \'fr\' or \'de\').')
    parser.add_argument('--max-words-langs', default=None, type=int, nargs="+",
                        help='Maximum number of words in each language dictionary (0 unlimited).')
    parser.add_argument('--encoder-langs', default=None, type=str, nargs="+",
                        help='Each language\'s encoder (either \'bow\', \'gru\' or \'lstm\').')
    parser.add_argument('--use-all-costs', default=False, action='store_true',
                        help='Whether to use costs between sentences in different languages in cost function or just costs between sentences and images.')
    parser.add_argument('--lambda-sent-sent', default=0.5, type=float,
                        help='Weight to use to scale sentence-to-sentence subcosts.')
    parser.add_argument('--lambda-img-sent',  default=0.5, type=float,
                        help='Weight to use to scale image-to-sentence subcosts.')

    parser.add_argument('--disp-freq', type=int, default=100, help='Display training statistics frequency (in model updates).')
    parser.add_argument('--valid-freq', type=int, default=100, help='Evaluate on valid set frequency (in model updates).')
    parser.add_argument('--test-freq', type=int, default=100, help='Evaluate on test set frequency (in model updates).')
    parser.add_argument('--attention-type', type=str, default='dot', choices=['dot', 'general'],
                        help='Attention type: one of \'dot\', \'general\'. Defaults to \'dot\'.')
    parser.add_argument('--decay-c-general-attention', type=float, default=0.0, help='L2 regularisation weight for the attention matrices.')
    args = parser.parse_args()
    assert(args.langs is not None)

    if args.debug:
        saveto=args.saveto+'.debug'
    else:
        saveto=args.saveto

    possible_models = [ 'f30k-translational',
                        'f30k-comparable-full',
                        'f30k-comparable-newsplits' ]

    lang_str=""
    for lang in args.langs:
        lang_str+=lang+"-"
    lang_str=lang_str[:-1] # remove trailing hyphen

    #saveto = default_saveto if args.saveto is None else args.saveto
    saveto = args.saveto

    if args.max_words_langs is None:
        args.max_words_langs = [0]*len(args.langs)

    # sanity checks
    assert(any([m in args.data for m in possible_models]))
    assert(args.maximum_sentence_length >= args.minimum_sentence_length)
    assert(args.margin >= 0. and args.margin <= 1.)
    assert(len(args.langs) >= 1)
    assert(len(args.langs) == len(args.max_words_langs) == len(args.encoder_langs))

    # Model options
    data            = args.data
    margin          = args.margin
    dim             = args.hidden_layer_dimensionality
    dim_multimodal  = args.multimodal_embeddings_dimensionality
    dim_image       = 4096
    dim_word        = args.word_embeddings_dimensionality
    max_epochs      = args.max_epochs
    dispFreq        = args.disp_freq
    decay_c         = args.decay_c
    grad_clip       = 1.
    maxlen_w        = args.maximum_sentence_length
    optimizer       = 'adam'
    batch_size      = args.batch_size
    validFreq       = args.valid_freq
    testFreq        = args.test_freq
    lrate           = 0.0002
    saveto          = saveto
    reload_         = args.reload
    debug           = args.debug
    use_dropout     = args.use_dropout
    #dropout_prob    = args.dropout_prob
    dropout_embedding = args.dropout_embedding
    dropout_hidden    = args.dropout_hidden
    dropout_source    = args.dropout_source
    attention_type    = args.attention_type
    decay_c_general_attention = args.decay_c_general_attention

    bidirectional         = args.bidirectional_encoder
    n_enc_hidden_layers   = args.n_encoder_hidden_layers
    create_dictionaries   = args.create_dictionaries
    dictionaries_min_freq = args.dictionaries_min_freq
    load_test             = args.load_test
    # multimodal, language dependent options
    langs           = args.langs
    encoders        = OrderedDict(zip(args.langs, args.encoder_langs))
    max_words_langs = OrderedDict(zip(args.langs, args.max_words_langs)) # default is 0 (unlimited)
    use_all_costs   = args.use_all_costs
    lambda_sent_sent= args.lambda_sent_sent
    lambda_img_sent = args.lambda_img_sent

    model_options = {}
    model_options['data'] = data
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
    #model_options['dropout_prob'] = dropout_prob
    model_options['dropout_embedding'] = dropout_embedding
    model_options['dropout_hidden'] = dropout_hidden
    model_options['dropout_source'] = dropout_source
    model_options['bidirectional_enc'] = bidirectional
    model_options['debug'] = debug
    model_options['n_enc_hidden_layers'] = n_enc_hidden_layers
    model_options['create_dictionaries'] = create_dictionaries
    model_options['dictionaries_min_freq'] = dictionaries_min_freq
    model_options['load_test'] = load_test
    model_options['attention_type'] = attention_type
    model_options['decay_c_general_attention'] = decay_c_general_attention
    # new multimodal embeddings parameters
    model_options['langs'] = langs
    model_options['encoders'] = encoders
    model_options['max_words'] = max_words_langs
    model_options['use_all_costs'] = use_all_costs
    model_options['lambda_sent_sent'] = lambda_sent_sent
    model_options['lambda_img_sent'] = lambda_img_sent

    print "Multilingual multimodal embeddings..."
    trainer(**model_options)
