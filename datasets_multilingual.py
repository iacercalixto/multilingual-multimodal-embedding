"""
Dataset loading
"""
import numpy
import os
import tables

def load_multilingual_dataset(path_to_data='./f30k-comparable-newsplits',
    langs=['en', 'de'], load_train=True, load_dev=True, load_test=True,
    load_images=True, image_features='vgg19.fc7'):
    """
    Load captions for languages in `langs` parameter and image features described in `image_features` param
    `image_features` only accepts value 'vgg19.fc7'
    """
    assert(type(langs) == list and len(langs)>=2)

    loc = path_to_data + "/"
    assert( os.path.isdir(loc) ), 'Invalid data folder: %s'%str(loc)
    
    # Captions
    train_caps_list = [None]*len(langs)
    dev_caps_list = [None]*len(langs)
    test_caps_list = [None]*len(langs)
    if load_train:
        for lang_idx, lang in enumerate(langs):
            train_caps_list[lang_idx] = []
            train_fname = loc+"train."+lang
            with open(train_fname, 'rb') as f:
                for line in f:
                    train_caps_list[lang_idx].append(line.strip())

    for lang_idx, lang in enumerate(langs):
        # dev
        if load_dev:
            dev_caps_list[lang_idx] = []
            dev_fname = loc+"dev."+lang
            with open(dev_fname, 'rb') as f:
                for line in f:
                    dev_caps_list[lang_idx].append(line.strip())
        # test
        if load_test:
            test_caps_list[lang_idx] = []
            test_fname = loc+"test."+lang
            with open(test_fname, 'rb') as f:
                for line in f:
                    test_caps_list[lang_idx].append(line.strip())
           
    # Image features
    if image_features=='vgg19.fc7':
        #train_ims = numpy.load(loc+"train_fc.npz")['arr_0'] if load_train and load_images else None
        #dev_ims  = numpy.load(loc+"dev_fc.npz")['arr_0'] if load_dev and load_images  else None
        #test_ims = numpy.load(loc+"test_fc.npz")['arr_0'] if load_test and load_images  else None
        train_file = tables.open_file(loc+"flickr30k_train_vgg19_bn_cnn_features.hdf5", mode='r') if load_train and load_images else None
        train_ims = train_file.root.global_feats[:]
        valid_file = tables.open_file(loc+"flickr30k_valid_vgg19_bn_cnn_features.hdf5", mode='r') if load_train and load_images else None
        dev_ims = valid_file.root.global_feats[:]
        test_file  = tables.open_file(loc+"flickr30k_test_vgg19_bn_cnn_features.hdf5", mode='r') if load_test and load_images  else None
        test_ims = test_file.root.global_feats[:]
        train_file.close()
        valid_file.close()
        test_file.close()
    else:
        raise Exception("image_features parameter value not supported: %s"%str(image_features))

    n_sentences_per_image = 5 if 'f30k-comparable' in path_to_data else 1
    print "n_sentences_per_image: %i"%int(n_sentences_per_image)

    return (train_caps_list, train_ims, n_sentences_per_image), \
            (dev_caps_list, dev_ims, n_sentences_per_image), \
            (test_caps_list, test_ims, n_sentences_per_image)

if __name__=="__main__":
    pass

