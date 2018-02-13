"""
Layers for multimodal-ranking
"""
import theano
import theano.tensor as tensor

import numpy

from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, linear

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer')
          }

def get_layer(name):
    """
    Return param init and feedforward functions for the given layer name
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# Feedforward layer
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    """
    Affine transformation + point-wise nonlinearity
    """
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = xavier_weight(nin, nout)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    """
    Feedforward pass
    """
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])

# dropout
def dropout_layer(state_before, use_noise, trng, prob=0.5):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=prob, n=1,
                                     dtype=state_before.dtype),
        state_before * prob)
    return proj

# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value):
    proj = tensor.switch(
        use_noise,
        trng.binomial(shape, p=value, n=1,
                                     dtype='float32'),
        theano.shared(numpy.float32(value)))
    return proj

# lstm layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None):
    """
    Init the LSTM parameters
    """
    assert(not nin is None and not dim is None)

    # input to hidden weights
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix, 'W')] = W
    # hidden to hidden (recurrent) weights
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    # biases
    b = numpy.zeros((4 * dim,))
    params[_p(prefix, 'b')] = b.astype(theano.config.floatX)

    return params

def lstm_layer(tparams, state_below, init_states, options, prefix='lstm', mask=None, dim=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    if dim==None:
        dim = options['dim']

    def _step(m_, x_, h_, c_, U, b):
        preact = tensor.dot(h_, U)
        preact += x_
        preact += b

        i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, dim))
        c =         tensor.tanh(_slice(preact, 3, dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    init_state_h = tensor.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, dim)
    init_state_c = tensor.alloc(numpy.asarray(0., dtype=theano.config.floatX), n_samples, dim)

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'b')]],
                                outputs_info=[init_state_h, init_state_c],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                strict=True)

    return rval


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    Gated Recurrent Unit (GRU)
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def gru_layer(tparams, state_below, options, init_state=None, prefix='gru', mask=None, one_step=False,
              emb_dropout=None,
              rec_dropout=None,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    if init_state is None:
        init_states = [tensor.alloc(0., n_samples, dim)]
    else:
        init_states = [init_state]

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*emb_dropout[0], tparams[_p(prefix, 'W')]) + \
        tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*emb_dropout[1], tparams[_p(prefix, 'Wx')]) + \
        tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux, rec_dropout):

        preact = tensor.dot(h_*rec_dropout[0], U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_*rec_dropout[1], Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')],
                   rec_dropout]

    if one_step:
        rval = _step(*(seqs+init_states+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=init_states,
                                    non_sequences=shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=False,
                                    strict=True)
    rval = [rval]
    return rval

