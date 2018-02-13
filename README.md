## Multilingual and multi-modal sentence embeddings

Code to train and use the **multilingual and multi-modal sentence embeddings** described in the paper [Sentence-Level Multilingual Multi-modal Embedding for Natural Language Processing](http://www.acl-bg.org/proceedings/2017/RANLP%202017/pdf/RANLP020.pdf).

Notice: this code-base is derived from the `visual-semantic embedding` project by Jamie Ryan Kiros [here](https://github.com/ryankiros/visual-semantic-embedding).


## Pre-requisites
Python 2.7
Theano 0.9.0
A recent version of [numpy](http://www.numpy.org/)

If you wish to extract image features, install [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) and Pytorch. If you are using anaconda with python3, you can install both Pytorch and the pretrained models by running the following commands:

```bash
conda install numpy pyyaml mkl setuptools cmake cffi
conda install -c pytorch magma-cuda80
pip install pretrainedmodels
```


## Getting started

First, download the [Multi30k data set](http://aclweb.org/anthology/W16-3210.pdf), available [here](http://www.statmt.org/wmt16/multimodal-task.html). Make sure you have the files `train_images.txt`, `val_images.txt` and `test_images.txt`, as well as all the image files inside a directory which we henceforth assume is `/path/to/m30k/flickr30k-images/`.

In order to extract global image features using a pre-trained VGG19 network (4098D features from the FC7 layer), first make sure the images are available (e.g. under `/path/to/m30k/flickr30k-images`), and that you have all the three files `train_images.txt`, `val_images.txt` and `test_images.txt`. Run the following command:

```bash
python extract_image_features.py
        --pretrained_cnn vgg19_bn
        --gpuid 1
        --images_path /path/to/m30k/flickr30k-images/
        --train_fnames /path/to/m30k/train_images.txt
        --valid_fnames /path/to/m30k/val_images.txt
        --test_fnames /path/to/m30k/test_images.txt
```

This will extract the global visual features for the training, dev and test sets into the project folder. The last step is to move the extracted features into the data set path:

```bash
mv flickr30k_*_*_cnn_features.hdf5 /path/to/m30k/
```

## Training a model

In order to train a model, first create the dictionaries that will be used. Assuming the Multi30k's training, validation and test sets (i.e. `train.en`, `train.de`, `dev.en`, `dev.de`, `test.en`, `test.de`) are available under the `/path/to/m30k/` directory, run:

```bash
python create_dictionaries.py --data /path/to/m30k/
```

This will create dictionaries in a default directory `./dicts/`. After you have created the dictionaries, open the `run_f30kC.sh` script and set the paths to the Multi30k data (e.g. `/path/to/m30k/` as above). Then, simply run the following:

```bash
./run_f30kC.sh
```

A model will be trained from scratch with default hyper-parameters. Please run `python do_train.py --help` to see all the options and hyper-parameters available.


## Evaluating a model

To evaluate a model, simply run:

```bash
python do_evaluate.py --data /path/to/m30k/ --saveto models/model-name.npz
```

By default, models are saved under the `./models/` directory. This will evaluate the model named `model-name.npz` in the tasks of image-sentence ranking in both directions (ranking sentences given images and vice-versa).


## Citations

If you found this code interesting and/or useful, please consider citing the following paper:

```
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
```

