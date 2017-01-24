# Deep Stochastic Radar Models

Supplementary material for _Deep Stochastic Radar Models_, by T. Wheeler, M. Holder, H. Winner, and M. Kochenderfer, submitted to IV 2017.

## Model Training Scripts

This repository contains the model training scripts, which define the model architectures and training procedure.
All scripts run in python 2.7 and train models using [Keras](https://github.com/fchollet/keras) with the [Tensorflow](https://www.tensorflow.org/) backend.

The scripts use the argparse package to allow for argument passing from the command line.
All scripts support:

| parameter  | type  | default | description |
|------------|-------|-------|---------------------------|
| batch_size | int   | 16    | the batch size            |
| nb_epoch   | int   | 100   | number of training epochs |
| verbosity  | int   | 1     | whether to be verbose     |
| train      | int   | 1  | whether to train the model |
| l2         | float | 0.001 | l2 weight regularization  |
| act        | str   | relu  | the activation function   |
| opt        | str   | adadelta  | the optimizer         |
| dset       | str   | dset_runway.h5  | path to the .h5 training dataset |
| tbdir      | str   | /tmp/keras_drm_MODELNAME/ | the tensorboard directory, used for log files based on execution time |
| tblog      | str   | '' | one can optionally specify the exact log filepath |
| tblog      | str   | '' | one can optionally specify the exact tensorboard log filepath |
| save       | str   | ''  | optional path to the save file containing the model weights |
| load       | str   | ''  | optional path to a file containing the model weights to load before training |
| save_pred  | str   | ''  | where to save model predictions |

Some parameters exist for specific models, such as `nrGaussians` for the GMM model.

