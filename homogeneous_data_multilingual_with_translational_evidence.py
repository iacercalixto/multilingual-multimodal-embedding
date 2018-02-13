import numpy
import copy
import sys
from itertools import chain

class HomogeneousDataMultilingualWithTranslationalEvidence():

    def __init__(self, data, batch_size=128, maxlen=None, minlen=None):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen

        self.prepare()
        self.reset()


    def prepare(self):
        self.caps = self.data[0] # image descriptions
        self.feats = self.data[1] # image features
        self.is_translational = self.data[2] # flag for translational vs. comparable minibatch

        # use first language as reference, "source" sentences
        self.caps_src = self.caps[0]
        #print len(self.caps_src)
        # find unique lengths
        self.lengths_src = [len(cc.split()) for cc in self.caps_src]
        self.len_unique_src = numpy.unique(self.lengths_src)
        # remove any overly long sentences
        if self.maxlen:
            self.len_unique_src = [ll for ll in self.len_unique_src if ll <= self.maxlen]
        # remove any overly short sentences
        if self.minlen:
            self.len_unique_src = [ll for ll in self.len_unique_src if ll >= self.minlen]

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

        return caps_ret, feats, self.is_translational

    def __iter__(self):
        return self

def prepare_data(caps, features, is_translational, worddicts, model_options, maxlen=None):
    """
    Put data into format usable by the model.

    caps            ::  list of lists. outer list has one element per language in model;
                        each inner list contain strings (sentence descriptions) in the appropriate language.
    features        ::  numpy array containing image features
    is_translational::  integet indicating translational vs. comparable minibatch (0 or 1)
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

            seqs[i].append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])
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

    if maxlen != None and (numpy.max(lengths_caps) >= maxlen):
        new_seqs = []
        new_feat_list = []
        new_lengths_caps = []

        filtered_out = []
        for idx, (ls, cs, y) in enumerate(zip(lengths_caps, zip(*seqs), feat_list)):
            if not all([l<maxlen for l in ls]):
                filtered_out.append(idx)

        print "filtered out idx: ", filtered_out 

        for idx, (ls, cs, y) in enumerate(zip(lengths_caps, zip(*seqs), feat_list)):
            if idx in filtered_out:
                continue

            new_seqs.append(cs)
            new_lengths_caps.append(ls)
            new_feat_list.append(y)
            #if all([l<maxlen for l in ls]):
            #    new_seqs.append(cs)
            #    new_lengths_caps.append(ls)
            #    new_feat_list.append(y)
            #else:
            #    lso, cso, yo = [], [], []
            #    # add only the ones that do not surpass maxlen
            #    for l_, c_ in zip(ls, cs):
            #        if l_>=maxlen:
            #            continue
            #        lso.append(l_)
            #        cso.append(c_)

            #    new_seqs.append(cso)
            #    new_lengths_caps.append(lso)
            #    new_feat_list.append(y)

        #seqs = new_seqs
        seqs = zip(*new_seqs)
        lengths_caps = new_lengths_caps
        feat_list = new_feat_list

        #n_samples, n_languages = len(seqs[0]), len(seqs)
        #print "seqs: ", seqs
        #print "lengths_caps: ", lengths_caps
        #print "n_samples: ", n_samples
        #print "n_languages: ", n_languages

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

