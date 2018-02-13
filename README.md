## Multilingual and multi-modal sentence embeddings

Code to train and use the **multilingual and multi-modal sentence embeddings** described in the paper [Sentence-Level Multilingual Multi-modal Embedding for Natural Language Processing](http://www.acl-bg.org/proceedings/2017/RANLP%202017/pdf/RANLP020.pdf).

Notice: this code-base is derived from the `visual-semantic embedding` project by Jamie Ryan Kiros [here](https://github.com/ryankiros/visual-semantic-embedding).

## Pre-requisites
Python 2.7
Theano 0.9.0
A recent version of [numpy](http://www.numpy.org/)

## Getting Started

First, you should download the [Multi30k data set](http://shannon.cs.illinois.edu/DenotationGraph/). In order to download the image features for this data set, extracted using a pre-trained VGG19 network (4098D features from the FC7 layer) and released by Kiros, run:

```bash
wget http://www.cs.toronto.edu/~rkiros/datasets/f30k.zip
```

## Training a model

In order to train a model, first create the dictionaries that will be used by the model:

```bash
python create_dictionaries.py --data /path/to/data/
```

This will create dictionaries in the default directory `./dicts/`. After you have created the dictionaries, open the `run_f30kC.sh` script and set the paths to the Multi30k datai (e.g. `/path/to/data/` as above). Then, simply run the following:

```bash
./run_f30kC.sh
```

A model will be trained from scratch with default hyper-parameters. Please run `python do_train.py --help` to see all the options and hyper-parameters available.

## Evaluating a model

To evaluate a model, simply run:

```bash
python do_evaluate.py --data /path/to/data/ --saveto /path/to/models/model-name.npz
```

By default, models are saved under the `./models/` directory. This will evaluate the model named `model-name.npz` in the tasks of image-sentence ranking.

## Citations

If you found this code interesting and/or useful, please consider citing the following paper:

@InProceedings{CalixtoLiu2017:RANLP,
  author    = {Calixto, Iacer and Liu, Qun},
  title     = {{Sentence-Level Multilingual Multi-modal Embedding for Natural Language Processing}},
  booktitle = {Proceedings of the International Conference Recent Advances in Natural Language Processing, RANLP 2017},
  month     = {September},
  year      = {2017},
  address   = {Varna, Bulgaria},
  pages     = {139--148},
  url       = {https://doi.org/10.26615/978-954-452-049-6_020}
}


