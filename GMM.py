from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.regularizers import l2, l1
from keras.layers import Input, Dense, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.optimizers import SGD

import time
import argparse
import tensorflow as tf
tf.python.control_flow_ops = tf # bugfix see https://github.com/fchollet/keras/issues/3857
import numpy as np
import random
import h5py

##########################
# Input Parser

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--nb_epoch',   type=int, default=100)
parser.add_argument('--l2',         type=float, default=0.0001) # weight l2 regularization
parser.add_argument('--verbosity',  type=int, default=1)
parser.add_argument('--train',      type=int, default=1)
parser.add_argument('--act',        type=str, default='relu') # activation
parser.add_argument('--opt',        type=str, default='adadelta') # optimizer
parser.add_argument('--tbdir',      type=str, default='/tmp/keras_drm_GMM/')
parser.add_argument('--tblog',      type=str, default='')
parser.add_argument('--dset',       type=str, default='dset_runway.h5')
parser.add_argument('--save',       type=str, default='') # log file
parser.add_argument('--load',       type=str, default='') # optionally specify filepath for initial net to load from h5 file
parser.add_argument('--save_pred',  type=str, default='gmm_predictions.h5')
parser.add_argument('--nrGaussians', type=int, default=2)
args = parser.parse_args()

if len(args.tblog) == 0:
	args.tblog = args.tbdir + time.strftime("run%Y%m%d-%H%M%S")

if args.verbosity > 0:
	print args

##########################
# Load Data

data = h5py.File(args.dset, 'r')
O_test = data['data/O_test'][:]
T_test = data['data/T_test'][:]
Y_test = data['data/Y_test'][:]
O_train = data['data/O_train'][:]
T_train = data['data/T_train'][:]
Y_train = data['data/Y_train'][:]
data.close()

O_train = np.swapaxes(np.swapaxes(O_train, 0, 3), 1, 2)
T_train = np.swapaxes(np.swapaxes(T_train, 0, 3), 1, 2)
Y_train = np.swapaxes(np.swapaxes(Y_train, 0, 3), 1, 2)
O_test = np.swapaxes(np.swapaxes(O_test, 0, 3), 1, 2)
T_test = np.swapaxes(np.swapaxes(T_test, 0, 3), 1, 2)
Y_test = np.swapaxes(np.swapaxes(Y_test, 0, 3), 1, 2)

n_train = O_train.shape[0] - (O_train.shape[0]%args.batch_size)
n_test = O_test.shape[0] - (O_test.shape[0]%args.batch_size)

O_train = O_train[0:n_train,:,:,:]
T_train = T_train[0:n_train,:,:,:]
Y_train = Y_train[0:n_train,:,:,:]
O_test = O_test[0:n_test,:,:,:]
T_test = T_test[0:n_test,:,:,:]
Y_test = Y_test[0:n_test,:,:,:]

if args.verbosity > 0:
	print "\nSHAPES"
	print "\tO_train shape: ", O_train.shape
	print "\tT_train shape: ", T_train.shape
	print "\tY_train shape: ", Y_train.shape
	print "\tO_test shape:  ", O_test.shape
	print "\tT_test shape:  ", T_test.shape
	print "\tY_test shape:  ", Y_test.shape
	print "\n"

# #########################
# Construct (& Load) Model

# Object Head
O_in = Input(shape=(O_train.shape[1], O_train.shape[2], O_train.shape[3]), dtype='float32', name='O')                          # [None, 2, 1,  9]
o_conv1 = Convolution2D(16, 1, 1, name='o_conv1', border_mode='same', activation=args.act, W_regularizer=l2(args.l2))(O_in)    # [None, 2, 1, 16]
o_conv2 = Convolution2D(16, 1, 1, name='o_conv2', border_mode='same', activation=args.act, W_regularizer=l2(args.l2))(o_conv1) # [None, 2, 1, 16]
o_flat  = Flatten()(o_conv2) # [None, 32]

##########################
# Terrain Head
#T_in = Input(shape=(T_train.shape[1],), dtype='float32', name='T') # [None, 1]

T_in = Input(shape=(T_train.shape[1],T_train.shape[2], T_train.shape[3]), dtype='float32', name='T') # [None, 1]
T_conv1 = Convolution2D(2, 3, 3, name='t_conv1', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2)) # [None, 64, 64, 1]
T_conv2 = Convolution2D(4, 3, 3, name='t_conv2', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2)) # [None, 32, 32, 2]
T_conv3 = Convolution2D(8, 3, 3, name='t_conv3', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2)) # [None, 16, 16, 4]
T_conv4 = Convolution2D(16, 3, 3, name='t_conv4', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2)) # [None, 8, 8, 8]
T_conv5 = Convolution2D(32, 3, 3, name='t_conv5', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2)) # [None, 4, 4, 16]
T_conv6 = Convolution2D(64, 3, 3, name='t_conv6', border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None, 2, 2, 32]
T_out   = Flatten() # [None, 32]

t_conv1 = T_conv1(T_in)
t_conv2 = T_conv2(t_conv1)
t_conv3 = T_conv3(t_conv2)
t_conv4 = T_conv4(t_conv3)
t_conv5 = T_conv5(t_conv4)
t_conv6 = T_conv6(t_conv5)
t_out = T_out(t_conv6)

##########################
# Dense Processing
merge_heads = Merge(mode='concat', concat_axis=1)([o_flat, t_out]) # [None, 33]
dense1 = Dense(128, name='dense1', activation=args.act, W_regularizer=l2(args.l2))(merge_heads)
dense2 = Dense(256, name='dense2', activation=args.act, W_regularizer=l2(args.l2))(dense1)

reshape2  = Reshape((8, 8, 4))(dense2) # [None, 8, 8, 2]
deco1_1   = Deconvolution2D(8, 3, 3, name='deco1_1', output_shape=(args.batch_size, 16, 16, 8), activation=args.act, border_mode='same', subsample=(2, 2), W_regularizer=l2(args.l2))(reshape2) # [None, 16, 16, 8]
deco1_2   = Deconvolution2D(8, 3, 3, name='deco1_2', output_shape=(args.batch_size, 16, 16, 8), activation=args.act, border_mode='same', subsample=(1, 1), W_regularizer=l2(args.l2))(deco1_1) # [None, 16, 16, 8]
deco2_1   = Deconvolution2D(8, 3, 3, name='deco2_1', output_shape=(args.batch_size, 32, 32, 8), activation=args.act, border_mode='same', subsample=(2, 2), W_regularizer=l2(args.l2))(deco1_2) # [None, 32, 32, 8]
deco2_2   = Deconvolution2D(8, 3, 3, name='deco2_2', output_shape=(args.batch_size, 32, 32, 8), activation=args.act, border_mode='same', subsample=(1, 1), W_regularizer=l2(args.l2))(deco2_1) # [None, 32, 32, 8]
deco3_1   = Deconvolution2D(8, 3, 3, name='deco3_1', output_shape=(args.batch_size, 64, 64, 8), activation=args.act, border_mode='same', subsample=(2, 2), W_regularizer=l2(args.l2))(deco2_2) # [None, 64, 64, 4]
Y_out     = Deconvolution2D(3*args.nrGaussians, 3, 3, name='Y', output_shape=(args.batch_size, 64, 64, 3*args.nrGaussians), activation='linear', border_mode='same', subsample=(1, 1), W_regularizer=l2(args.l2))(deco3_1) # [None, 64, 64, 2]

def neg_logl(Y_true, Y_pred):
	weights = tf.add(0.0001,tf.square(Y_pred[:,:,:,2*args.nrGaussians:3*args.nrGaussians])) # 4:6
	weights = tf.div(weights, tf.expand_dims(tf.reduce_sum(weights, reduction_indices=3), 3))
	y       =        K.flatten(        tf.tile(Y_true,[1,1,1,args.nrGaussians]))
	mean    =        K.flatten(        Y_pred[:,:,:,0*args.nrGaussians:1*args.nrGaussians] )
	var  	= tf.add(K.flatten( K.relu(Y_pred[:,:,:,1*args.nrGaussians:2*args.nrGaussians])), 0.001)  # ensures it is positive
	W       =        K.flatten(       weights[:,:,:,:])
	logl 	= -0.5*K.mean(W * (K.log(var) + K.square(mean - y)/var), axis=-1)

	return -logl

model = Model(input=[O_in, T_in], output=Y_out)
model.compile(optimizer=args.opt, loss=neg_logl)

if len(args.load) > 0:
	if args.verbosity > 0:
		print 'loading model weights from ', args.load
	model.load_weights(args.load)

if args.verbosity > 0:
	model.summary()

##########################
# Training

if args.train == 1:
	model.fit([O_train, T_train], Y_train,
          nb_epoch=args.nb_epoch,
          batch_size=args.batch_size,
          shuffle=True,
          verbose=args.verbosity,
          validation_data=([O_test, T_test], Y_test),
          callbacks=[TensorBoard(log_dir=args.tblog),
          ModelCheckpoint('training_GMM.h5', save_best_only=True, save_weights_only=True),
          EarlyStopping(monitor='val_loss',patience=args.nb_epoch/2)],
          )


##########################
# Save Model

if len(args.save) > 0:
	if args.verbosity > 0:
		print 'saving model weights to ', args.save
	model.save_weights(args.save)

if len(args.save_pred) > 0:
	print 'Saving predictions'
	f = h5py.File(args.save_pred, 'w')
	f.create_dataset("predict/Y_pred", data=model.predict([O_test, T_test], batch_size=args.batch_size, verbose=args.verbosity))
	f.create_dataset("predict/Y_pred_train", data=model.predict([O_train, T_train], batch_size=args.batch_size, verbose=args.verbosity))
	f.close()
