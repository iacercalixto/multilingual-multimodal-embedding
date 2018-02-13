#!/bin/bash

#WSDIR="/media/storage4tb/tools/multilingual-multimodal-sentence-embeddings"
WSDIR=$(pwd)
MODELDIR="$WSDIR/models"
LOGDIR="$WSDIR/logs"
PYTHON=$HOME/anaconda2/bin/python
device="gpu"
DATADIR='/media/storage2tb/resources/multilingual-multimodal/f30k-comparable-newsplits'

# run job
cd $WSDIR

batch_size=128
let updates_per_minibatch=145000/$batch_size
let disp_freq=$updates_per_minibatch/2  # every 1/2 epoch
valid_freq=$updates_per_minibatch       # every 1 epoch
test_freq=$updates_per_minibatch        # every 1 epoch

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn \
        python do_train.py \
        --data $DATADIR \
        --saveto "$MODELDIR/f30k-comparable-newsplits.no-dropout-lambda-sent-sent-0.0-lambda-img-sent-1.0.1st-run.npz" \
        --langs en de \
        --encoder-langs gru gru \
        --margin 0.2 \
        --max-epochs 200 \
        --hidden-layer-dimensionality 1024 \
        --multimodal-embeddings-dimensionality 2048 \
        --n-encoder-hidden-layers 1 \
        --lambda-sent-sent 0.0 \
        --lambda-img-sent 1.0 \
        --batch-size $batch_size \
        --reload \
        --create-dictionaries \
        --disp-freq $disp_freq \
        --valid-freq $valid_freq \
        --test-freq $valid_freq \
        --load-test \
        2>&1 | tee -a $LOGDIR/f30k-comparable-newsplits.no-dropout-lambda-sent-sent-0.0-lambda-img-sent-1.0.1st-run.log

#--use-all-costs \
#--use-dropout \
#--dropout-prob 0.3 \


