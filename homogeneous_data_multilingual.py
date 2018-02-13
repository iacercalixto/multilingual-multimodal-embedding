import numpy
import copy
import sys
from itertools import chain

class HomogeneousDataMultilingual():

    def __init__(self, data, batch_size=128, maxlen=None):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        # default is one sentence per image
        self.n_sentences_per_image = int(self.data[2]) if len(self.data)==3 else 1
        print "number of sentences per image: %i"%(self.n_sentences_per_image)

        self.prepare()
        self.reset()


    def prepare(self):
        self.caps = self.data[0] # image descriptions
        self.feats = numpy.repeat(self.data[1], self.n_sentences_per_image, axis=0) # image features

        # remove any overly long sentences
        if self.maxlen:
            self.caps[0] = [" ".join(cc.split()[:self.maxlen]) for cc in self.caps[0]]
            self.caps[1] = [" ".join(cc.split()[:self.maxlen]) for cc in self.caps[1]]

        # find unique lengths - using source as default
        self.lengths_src = [len(cc.split()) for cc in self.caps[0]]
        self.len_unique_src = numpy.unique(self.lengths_src)

        # indices of unique lengths
        self.len_indices_src = dict()
        self.len_counts_src = dict()
        for ll in self.len_unique_src:
            self.len_indices_src[ll] = numpy.where(self.lengths_src == ll)[0]
            self.len_counts_src[ll] = len(self.len_indices_src[ll])

        # current counter
        self.len_curr_counts_src = copy.copy(self.len_counts_src)


    def reset(self):
        self.len_curr_counts_src = copy.copy(self.len_counts_src)
        self.len_unique_src = numpy.random.permutation(self.len_unique_src)
        self.len_indices_pos_src = dict()

        for ll in self.len_unique_src:
            self.len_indices_pos_src[ll] = 0
            self.len_indices_src[ll] = numpy.random.permutation(self.len_indices_src[ll])

        self.len_idx_src = -1

    def next(self):
        count = 0
        while True:
            self.len_idx_src = numpy.mod(self.len_idx_src+1, len(self.len_unique_src))
            if self.len_curr_counts_src[self.len_unique_src[self.len_idx_src]] > 0:
                break
            count += 1
            if count >= len(self.len_unique_src):
                break
        if count >= len(self.len_unique_src):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts_src[self.len_unique_src[self.len_idx_src]])
        curr_pos = self.len_indices_pos_src[self.len_unique_src[self.len_idx_src]]
        # get the indices for the current batch
        curr_indices = self.len_indices_src[self.len_unique_src[self.len_idx_src]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos_src[self.len_unique_src[self.len_idx_src]] += curr_batch_size
        self.len_curr_counts_src[self.len_unique_src[self.len_idx_src]] -= curr_batch_size

        caps_ret = []
        for caps in self.caps:
            caps_ret.append([caps[ii] for ii in curr_indices] )

        feats = [self.feats[ii] for ii in curr_indices]

        return caps_ret, feats

    def __iter__(self):
        return self

def prepare_data(caps, features, worddicts, model_options, maxlen=None):
    """
    Put data into format usable by the model.

    caps            ::  list of lists. outer list has one element per language in model;
                        each inner list contain strings (sentence descriptions) in the appropriate language.
    features        ::  numpy array containing image features
    worddicts       ::  list of dictionaries. there is one dictionary for each language.
    model_options   ::  dictionary with model parameters.
    maxlen          ::  maximum length of a sentence (if larger sentence will be discarded for training).
    """
    langs = model_options['langs']
    assert(len(langs) == len(caps) == len(worddicts))

    seqs = []
    for _ in caps:
        seqs.append([])
    feat_list = []

    for n_samples, cc_s in enumerate(zip(*caps)):
        for i,cc in enumerate(cc_s):
            worddict = worddicts[i]
            lang = langs[i]
            # maximum unique words per language
            n_words = model_options["n_words_%s"%lang]

            seqs[i].append([worddict[w] if w in worddict and worddict[w] < n_words else 1 for w in cc.split()])
            #seqs[i].append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])

        feat_list.append(features[n_samples])

    # outermost list indexes unique sentences, innermost list indexes language.
    # a list of lists.
    lengths_caps = []
    for seq in seqs:
        for i,s in enumerate(seq):
            try:
                _ = lengths_caps[i]
            except:
                lengths_caps.append([])
            lengths_caps[i].append(len(s))
    #lengths_caps = [len(s) for seq in seqs for s in seq]

    y = numpy.zeros((len(feat_list), len(feat_list[0]))).astype('float32')
    for idx, ff in enumerate(feat_list):
        y[idx,:] = ff

    n_samples, n_languages = len(seqs[0]), len(seqs)
    maxlens = [numpy.max(lengths_caps)+1]*n_samples

    xs = []
    xmasks = []
    for lang_idx, (ss, ll) in enumerate(zip(seqs, zip(*lengths_caps))):
        x = numpy.zeros((maxlens[0], n_samples)).astype('int64')
        x_mask = numpy.zeros((maxlens[0], n_samples)).astype('float32')

        for idx, (s, l) in enumerate(zip(ss, ll)):
            x[:l,idx] = s
            x_mask[:l+1,idx] = 1.

        xs.append(x)
        xmasks.append(x_mask)

    assert( len(xs) == len(xmasks) == n_languages )

    return xs, xmasks, y

