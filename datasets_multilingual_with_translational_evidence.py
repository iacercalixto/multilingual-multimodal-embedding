"""
Dataset loading
"""
import numpy

#-----------------------------------------------------------------------------#
# Specify dataset(s) location here
#-----------------------------------------------------------------------------#
#path_to_data = '/ichec/work/dcu01/icalixto/resources/multilingual-multimodal/'
path_to_data = '/media/storage2tb/resources/multilingual-multimodal/'
#-----------------------------------------------------------------------------#

def load_multilingual_dataset(names=['f30k-comparable-newsplits', 'f30k-translational'],
        langs=['en', 'de'], load_train=True, load_dev=True, load_test=True):
    """
    Load comparable and translational descriptions
    in languages in `langs` parameter and image features
    """
    assert(type(langs) == list and len(langs)>=2), 'Must provide at least two languages'
    assert(type(names) == list and len(names)==2), \
        'Must provide one comparable and one translational dataset'

    loc_comparable = path_to_data + names[0] + "/"
    loc_translational = path_to_data + names[1] + "/"
    
    # Image descriptions
    # comparable
    c_train_caps_list = [None]*len(langs)
    c_dev_caps_list = [None]*len(langs)
    c_test_caps_list = [None]*len(langs)
    # translational
    t_train_caps_list = [None]*len(langs)
    t_dev_caps_list = [None]*len(langs)
    t_test_caps_list = [None]*len(langs)
    for lang_idx, lang in enumerate(langs):
        if load_train:
            c_train_caps_list[lang_idx] = []
            train_fname = loc_comparable+"train."+lang
            with open(train_fname, 'rb') as f:
                for line in f:
                    c_train_caps_list[lang_idx].append(line.strip())

            t_train_caps_list[lang_idx] = []
            train_fname = loc_translational+"train."+lang
            with open(train_fname, 'rb') as f:
                for line in f:
                    t_train_caps_list[lang_idx].append(line.strip())

        # dev
        if load_dev:
            c_dev_caps_list[lang_idx], t_dev_caps_list[lang_idx] = [], []
            #c_dev_caps_list[lang_idx], c_test_caps_list[lang_idx] = [], []
            #t_dev_caps_list[lang_idx], t_test_caps_list[lang_idx] = [], []
            dev_fname = loc_comparable+"dev."+lang
            with open(dev_fname, 'rb') as f:
                for line in f:
                    c_dev_caps_list[lang_idx].append(line.strip())
            dev_fname = loc_translational+"dev."+lang
            with open(dev_fname, 'rb') as f:
                for line in f:
                    t_dev_caps_list[lang_idx].append(line.strip())

        # test
        if load_test:
            c_test_caps_list[lang_idx], t_test_caps_list[lang_idx] = [], []
            try:
                test_fname = loc_comparable+"test."+lang
                with open(test_fname, 'rb') as f:
                    for line in f:
                        c_test_caps_list[lang_idx].append(line.strip())
                test_fname = loc_translational+"test."+lang
                with open(test_fname, 'rb') as f:
                    for line in f:
                        t_test_caps_list[lang_idx].append(line.strip())
            except:
                print "Could not load test set."

           
    # Image features
    #if name.startswith('f30k'):
    c_train_ims = numpy.load(loc_comparable+"train.npz")['arr_0'] if load_train else None
    c_dev_ims  = numpy.load(loc_comparable+"dev.npz")['arr_0'] if load_dev else None
    c_test_ims = numpy.load(loc_comparable+"test.npz")['arr_0'] if load_test else None
    t_train_ims = numpy.load(loc_translational+"train.npz")['arr_0'] if load_train else None
    t_dev_ims  = numpy.load(loc_translational+"dev.npz")['arr_0'] if load_dev else None
    t_test_ims = numpy.load(loc_translational+"test.npz")['arr_0'] if load_test else None
    #else:
    #    train_ims = numpy.load(loc+"train.npy") if load_train else None
    #    dev_ims  = numpy.load(loc+"dev.npy") if load_dev else None
    #    test_ims = numpy.load(loc+"test.npy") if load_test else None

    return ((c_train_caps_list, c_train_ims, 0), (t_train_caps_list, t_train_ims, 1)), \
           ((c_dev_caps_list, c_dev_ims, 0), (t_dev_caps_list, t_dev_ims, 1)), \
           ((c_test_caps_list, c_test_ims, 0), (t_test_caps_list, t_test_ims, 1))

