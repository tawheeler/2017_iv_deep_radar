# run: tensorboard --logdir=/tmp/keras_drm_vae

# export results:
# python keras_deep_vae.py --train 0 --dset /media/tim/Tim\ 1500\ GB/GriesheimDriveAround/dset_runway_final.h5 --load /media/tim/Tim\ 1500\ GB/2017_ivs_drm/models/vae_weights_complete.h5 --save_pred /media/tim/Tim\ 1500\ GB/2017_ivs_drm/models/pred_vae.h5

# NOTE: this sys stuff is there to prevent the "Using Tensorflow backend" from being constantly displayed
# import sys
# stderr = sys.stderr
# sys.stderr = open('/dev/null', 'w')

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.regularizers import l2, l1
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

# sys.stderr = stderr

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
parser.add_argument('--tbdir',      type=str, default='/tmp/keras_drm_vae/')
parser.add_argument('--tblog',      type=str, default='')
parser.add_argument('--dset',       type=str, default='dset_runway.h5')
parser.add_argument('--save',       type=str, default='') # log file
parser.add_argument('--load',       type=str, default='') # optionally specify filepath for initial net to load from h5 file
parser.add_argument('--save_pred',  type=str, default='vae_predictions.h5')

parser.add_argument('--logP_var', type=float, default=0.1)
parser.add_argument('--epsilon_std', type=float, default=0.01)
parser.add_argument('--dim_z', type=int, default=16)
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

##########################
# Object Head
O_in = Input(shape=(O_train.shape[1], O_train.shape[2], O_train.shape[3]), dtype='float32', name='O')                 # [None, 2, 1,  9]
O_conv1 = Convolution2D(16, 1, 1, name='o_conv1', border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None, 2, 1, 16]
O_conv2 = Convolution2D(16, 1, 1, name='o_conv2', border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None, 2, 1, 16]
O_out   = Flatten() # [None, 32]

o_conv1 = O_conv1(O_in)
o_conv2 = O_conv2(o_conv1)
o_out = O_out(o_conv2)

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
# X Head
#X_in = Merge(mode='concat', concat_axis=1)([o_out, T_in]) # [None, 33]
X_in = Merge(mode='concat', concat_axis=1)([o_out, t_out]) # [None, 33]
X_dense1 = Dense(128, name='x_dense1', activation=args.act, W_regularizer=l2(args.l2))
X_bnorm1 = BatchNormalization(mode=2)
X_dense2 = Dense(128, name='x_dense2', activation=args.act, W_regularizer=l2(args.l2))
X_out    = BatchNormalization(mode=2)

x_dense1 = X_dense1(X_in)
x_bnorm1 = X_bnorm1(x_dense1)
x_dense2 = X_dense2(x_bnorm1)
x_out = X_out(x_dense2)

##########################
# Y Head
Y_in = Input(shape=(Y_train.shape[1], Y_train.shape[2], Y_train.shape[3]), dtype='float32', name='Y_in') # [None, 64, 64, 1]
Y_conv1_1 = Convolution2D( 8, 3, 3, name='y_conv1_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None, 32, 32,  8]
Y_conv1_2 = Convolution2D( 8, 3, 3, name='y_conv1_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None, 32, 32,  8]
Y_conv2_1 = Convolution2D(16, 3, 3, name='y_conv2_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None, 16, 16, 16]
Y_conv2_2 = Convolution2D(16, 3, 3, name='y_conv2_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None, 16, 16, 16]
Y_conv3_1 = Convolution2D(32, 3, 3, name='y_conv3_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None,  8,  8, 32]
Y_conv3_2 = Convolution2D(32, 3, 3, name='y_conv3_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None,  8,  8, 32]
Y_conv4_1 = Convolution2D(32, 3, 3, name='y_conv4_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None,  4,  4, 32]
Y_conv4_2 = Convolution2D( 4, 3, 3, name='y_conv4_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)) # [None,  4,  4, 4]
Y_out     = Flatten() # [64]

y_conv1_1 = Y_conv1_1(Y_in)
y_conv1_2 = Y_conv1_2(y_conv1_1)
y_conv2_1 = Y_conv2_1(y_conv1_2)
y_conv2_2 = Y_conv2_2(y_conv2_1)
y_conv3_1 = Y_conv3_1(y_conv2_2)
y_conv3_2 = Y_conv3_2(y_conv3_1)
y_conv4_1 = Y_conv4_1(y_conv3_2)
y_conv4_2 = Y_conv4_2(y_conv3_2)
y_out = Y_out(y_conv4_2)

##########################
# Encoder
E_in = Merge(mode='concat', concat_axis=1)
E_dense1 = Dense(32, name='e_dense1', activation='linear', W_regularizer=l2(args.l2))
E_bnorm1 = BatchNormalization(mode=2)
E_dense2 = Dense(32, name='e_dense2', activation='linear', W_regularizer=l2(args.l2))
E_bnorm2 = BatchNormalization(mode=2)
Z_mean = Dense(args.dim_z, name='dense_z_mean', activation='linear', W_regularizer=l2(args.l2))
Z_logvar = Dense(args.dim_z, name='dense_z_logvar', activation='linear', W_regularizer=l2(args.l2))

e_in = E_in([x_out, y_out])
e_dense1 = E_dense1(e_in)
e_bnorm1 = E_bnorm1(e_dense1)
e_dense2 = E_dense2(e_bnorm1)
e_bnorm2 = E_bnorm2(e_dense2)
z_mean = Z_mean(e_bnorm2)
z_logvar = Z_logvar(e_bnorm2)

def sampling(sampling_args):
    z_mean, z_logvar = sampling_args
    epsilon = K.random_normal(shape=(args.batch_size, args.dim_z),
                              mean=0.0, std=args.epsilon_std)
    return z_mean + K.exp(z_logvar) * epsilon

Z = Lambda(sampling, output_shape=(args.dim_z,))
z = Z([z_mean, z_logvar])

# Decoder

D_in = Merge(mode='concat', concat_axis=1)
D_dense1  = Dense(128, name='dense3', activation=args.act, W_regularizer=l2(args.l2))
D_bnorm1  = BatchNormalization(mode=2)
D_dense2  = Dense(256, name='dense4', activation=args.act, W_regularizer=l2(args.l2))
D_bnorm2  = BatchNormalization(mode=2)
D_reshape = Reshape((8, 8, 4)) # [None, 8, 8, 2]
Deco1_1   = Deconvolution2D(8, 3, 3, name='deco1_1', output_shape=(args.batch_size, 16, 16, 8), activation=args.act, border_mode='same', subsample=(2, 2), W_regularizer=l2(args.l2)) # [None, 16, 16, 8]
Deco1_2   = Deconvolution2D(8, 3, 3, name='deco1_2', output_shape=(args.batch_size, 16, 16, 8), activation=args.act, border_mode='same', subsample=(1, 1), W_regularizer=l2(args.l2)) # [None, 16, 16, 8]
Deco2_1   = Deconvolution2D(8, 3, 3, name='deco2_1', output_shape=(args.batch_size, 32, 32, 8), activation=args.act, border_mode='same', subsample=(2, 2), W_regularizer=l2(args.l2)) # [None, 32, 32, 8]
Deco2_2   = Deconvolution2D(8, 3, 3, name='deco2_2', output_shape=(args.batch_size, 32, 32, 8), activation=args.act, border_mode='same', subsample=(1, 1), W_regularizer=l2(args.l2)) # [None, 32, 32, 8]
Deco3_1   = Deconvolution2D(4, 3, 3, name='deco3_1', output_shape=(args.batch_size, 64, 64, 4), activation=args.act, border_mode='same', subsample=(2, 2), W_regularizer=l2(args.l2)) # [None, 64, 64, 4]
Y_out     = Deconvolution2D(1, 3, 3, name='deco3_2',       output_shape=(args.batch_size, 64, 64, 1), activation='linear', border_mode='same', subsample=(1, 1), W_regularizer=l2(args.l2)) # [None, 64, 64, 1]

d_in = D_in([x_out, z])
d_dense1 = D_dense1(d_in)
d_bnorm1 = D_bnorm1(d_dense1)
d_dense2 = D_dense2(d_bnorm1)
d_bnorm2 = D_bnorm2(d_dense2)
d_reshape = D_reshape(d_bnorm2)
deco1_1 = Deco1_1(d_reshape)
deco1_2 = Deco1_2(deco1_1)
deco2_1 = Deco2_1(deco1_2)
deco2_2 = Deco2_2(deco2_1)
deco3_1 = Deco3_1(deco2_2)
y_out = Y_out(deco3_1)

def vae_loss(Y_true, Y_pred):
	y_true = K.flatten(Y_true)
	y_mean = K.flatten(Y_pred)

	logP_loss = 0.5 * K.mean(K.log(args.logP_var) + K.square(y_mean - y_true)/args.logP_var, axis=-1) # negative logl
	kl_loss =  -0.5 * K.mean(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1) # penalizes deviation from normal distr
	return logP_loss + kl_loss

model = Model(input=[O_in, T_in, Y_in], output=y_out)
model.compile(optimizer=args.opt, loss=vae_loss)

if len(args.load) > 0:
	if args.verbosity > 0:
		print 'loading model weights from ', args.load
	model.load_weights(args.load)

if args.verbosity > 0:
	model.summary()

##########################
# Training

if args.train != 0:
	model.fit([O_train, T_train, Y_train], Y_train,
          nb_epoch=args.nb_epoch,
          batch_size=args.batch_size,
          shuffle=True,
          verbose=args.verbosity,
          validation_data=([O_test, T_test, Y_test], Y_test),
          callbacks=[TensorBoard(log_dir=args.tblog),
                     ModelCheckpoint('training_vae.h5', save_best_only=True, save_weights_only=True),
                     EarlyStopping(monitor='val_loss',patience=args.nb_epoch/4)],
          )

##########################
# Save Model

if len(args.save) > 0:
	if args.verbosity > 0:
		print 'saving model weights to ', args.save
	model.save_weights(args.save)

if len(args.save_pred) > 0:

	print("SAVING PREDICTIONS")

	pred_o_conv1 = O_conv1(O_in)
	pred_o_conv2 = O_conv2(pred_o_conv1)
	pred_o_out = O_out(pred_o_conv2)

	pred_t_conv1 = T_conv1(T_in)
	pred_t_conv2 = T_conv2(pred_t_conv1)
	pred_t_conv3 = T_conv3(pred_t_conv2)
	pred_t_conv4 = T_conv4(pred_t_conv3)
	pred_t_conv5 = T_conv5(pred_t_conv4)
	pred_t_conv6 = T_conv6(pred_t_conv5)
	pred_t_out = T_out(t_conv6)


	pred_x_in = Merge(mode='concat', concat_axis=1)([pred_o_out, pred_t_out])
	pred_x_dense1 = X_dense1(pred_x_in)
	pred_x_bnorm1 = X_bnorm1(pred_x_dense1)
	pred_x_dense2 = X_dense2(pred_x_bnorm1)
	pred_x_out = X_out(pred_x_dense2)

	Z_test = np.random.randn(O_test.shape[0],args.dim_z)*args.epsilon_std
	Z_train = np.random.randn(O_train.shape[0],args.dim_z)*args.epsilon_std

	Z_in = Input(shape=(args.dim_z,), dtype='float32', name='pred_z') # drawn from N(O,I)
	pred_d_in = Merge(mode='concat', concat_axis=1)([pred_x_out, Z_in])
	pred_d_dense1 = D_dense1(pred_d_in)
	pred_d_bnorm1 = D_bnorm1(pred_d_dense1)
	pred_d_dense2 = D_dense2(pred_d_bnorm1)
	pred_d_bnorm2 = D_bnorm2(pred_d_dense2)
	pred_d_reshape = D_reshape(pred_d_bnorm2)
	pred_deco1_1 = Deco1_1(pred_d_reshape)
	pred_deco1_2 = Deco1_2(pred_deco1_1)
	pred_deco2_1 = Deco2_1(pred_deco1_2)
	pred_deco2_2 = Deco2_2(pred_deco2_1)
	pred_deco3_1 = Deco3_1(pred_deco2_2)
	pred_y_out = Y_out(pred_deco3_1)

	model = Model(input=[O_in, T_in, Z_in], output=pred_y_out)
	model.compile(optimizer=args.opt, loss=vae_loss)

	if len(args.load) > 0:
		if args.verbosity > 0:
			print 'loading model weights from ', args.load
		model.load_weights(args.load, by_name=True)

	#data = h5py.File(args.dset, 'r')
	#Z_pred = data['data/Z_pred'][:]
	#data.close()
	#Z_pred = np.swapaxes(Z_pred, 0, 1)
	#Z_pred = Z_pred[0:n_test,:]

	#if args.verbosity > 0:
	#		print "\tZ_pred shape:  ", Z_pred.shape

	f = h5py.File(args.save_pred, 'w')
	f.create_dataset("predict/Y_pred", data=model.predict([O_test, T_test, Z_test], batch_size=args.batch_size, verbose=args.verbosity))
	f.create_dataset("predict/Y_pred_train", data=model.predict([O_train, T_train, Z_train], batch_size=args.batch_size, verbose=args.verbosity))
	f.close()
