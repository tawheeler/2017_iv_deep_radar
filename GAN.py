from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Merge
from keras.layers.convolutional import Convolution2D, Deconvolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

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
parser.add_argument('--nsteps', type=int, default=10)
parser.add_argument('--l2',         type=float, default=0.0001) # weight l2 regularization
parser.add_argument('--verbosity',  type=int, default=1)
parser.add_argument('--train',      type=int, default=1)
parser.add_argument('--act',        type=str, default='relu') # activation
parser.add_argument('--opt',        type=str, default='adadelta') # optimizer
parser.add_argument('--tbdir',      type=str, default='/tmp/keras_drm_GAN/')
parser.add_argument('--tblog',      type=str, default='')
parser.add_argument('--dset',       type=str, default='dset_runway.h5')
parser.add_argument('--save',       type=str, default='') # log file
parser.add_argument('--load',       type=str, default='') # optionally specify filepath for initial net to load from h5 file
parser.add_argument('--save_pred',  type=str, default='gan_predictions.h5')

parser.add_argument('--fraction_loss_gan', type=float, default=0.95)
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


# ######################################
# Construct Discriminator Model

discriminator = Sequential()
discriminator.add(Convolution2D( 8, 3, 3, name='d_conv1_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2), input_shape=(64,64,1)))
discriminator.add(Convolution2D( 8, 3, 3, name='d_conv1_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(Convolution2D(16, 3, 3, name='d_conv2_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(Convolution2D(16, 3, 3, name='d_conv2_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(Convolution2D(32, 3, 3, name='d_conv3_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(Convolution2D(32, 3, 3, name='d_conv3_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(Convolution2D(32, 3, 3, name='d_conv4_1', subsample=(2,2), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(Convolution2D( 4, 3, 3, name='d_conv4_2', subsample=(1,1), border_mode='same', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(Flatten())
discriminator.add(Dense(16, name='d_dense1', activation=args.act, W_regularizer=l2(args.l2)))
discriminator.add(BatchNormalization(mode=2))
discriminator.add(Dense( 1, name='d_dense2', activation='sigmoid', W_regularizer=l2(args.l2)))

discriminator.compile(loss='binary_crossentropy', optimizer=args.opt)

# ##################################################
# Construct combined Generator - Discriminator Model

# Untrainable Discriminator
untrainable_discriminator = Sequential()
untrainable_discriminator.trainable=False
untrainable_discriminator.add(discriminator)

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
D_out     = Deconvolution2D(1, 3, 3, name='deco3_2', output_shape=(args.batch_size, 64, 64, 1), activation='linear', border_mode='same', subsample=(1, 1), W_regularizer=l2(args.l2)) # [None, 64, 64, 1]

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
d_out = D_out(deco3_1)
#print 'd_out.shape ', d_out.get_shape()

def vae_loss(Y_true, Y_pred):
	y_true = K.flatten(Y_true)
	y_mean = K.flatten(Y_pred)

	logP_loss = 0.5 * K.mean(K.log(args.logP_var) + K.square(y_mean - y_true)/args.logP_var, axis=-1) # negative logl
	kl_loss =  -0.5 * K.mean(1 + z_logvar - K.square(z_mean) - K.exp(z_logvar), axis=-1) # penalizes deviation from normal distr
	return logP_loss + kl_loss

# ######################################
# Construct Discriminator Model

adversary = Sequential()
adversary.add(Convolution2D(2, 3, 3, input_shape=(Y_train.shape[1], Y_train.shape[2], Y_train.shape[3]), name='a_conv1', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2))) # [None, 64, 64, 1]
adversary.add(Convolution2D(4, 3, 3,  name='a_conv2', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2))) # [None, 32, 32, 2]
adversary.add(Convolution2D(8, 3, 3,  name='a_conv3', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2))) # [None, 16, 16, 4]
adversary.add(Convolution2D(16, 3, 3, name='a_conv4', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2))) # [None, 8, 8, 8]
adversary.add(Convolution2D(32, 3, 3, name='a_conv5', border_mode='same', subsample=(2,2), activation=args.act, W_regularizer=l2(args.l2))) # [None, 4, 4, 16]
adversary.add(Convolution2D(1, 3, 3,  name='a_conv6', border_mode='same', subsample=(2,2), activation='sigmoid', W_regularizer=l2(args.l2))) # [None, 1, 1, 1]
adversary.add(Flatten()) # [None, 32]
adversary.compile(loss='binary_crossentropy', optimizer=args.opt, metrics =['binary_accuracy'])

# Untrainable Adversary
untrainable_adversary = Sequential()
untrainable_adversary.trainable=False
untrainable_adversary.add(adversary)

# Construct Generator

adv = untrainable_adversary(d_out)
generator = Model(input=[O_in, T_in, Y_in], output=[d_out, adv])
generator.compile(optimizer=args.opt,
				  loss=[vae_loss, 'binary_crossentropy'],
	              loss_weights=[1-args.fraction_loss_gan, args.fraction_loss_gan])

# ######################################
# Load from File

if len(args.load) > 0:
	if args.verbosity > 0:
		print 'loading model weights from ', args.load
	generator.load_weights(args.load)

if args.verbosity > 0:
	generator.summary()

##########################
# Training

if args.train == 1:

	# when training the discriminator, first half are realdata (class 1), second half predicted data (class 0)
	d_disc = np.vstack([np.ones((args.batch_size,1),dtype=np.float),np.zeros((args.batch_size,1),dtype=np.float)])

	# when training the generator, we want to fake out the discriminator, so want it to claim class 1
	d_comb = np.ones((args.batch_size,1), dtype=np.float)

	def train_discriminator(nsteps):
		mean_loss = 0.0
		for i in range(1,nsteps):
			# pick real samples
			batch_indeces = np.random.randint(0,O_train.shape[0],args.batch_size)
			y_real = Y_train[batch_indeces,:,:,:]

			# pick fake samples
			batch_indeces = np.random.randint(0,O_train.shape[0],args.batch_size)
			o_in = O_train[batch_indeces,:,:,:]
			t_in = T_train[batch_indeces,:,:,:]
			y_in = Y_train[batch_indeces,:,:,:]
			y_fake = generator.predict([o_in, t_in, y_in])[0]

			# train
			y_disc = np.vstack([y_real, y_fake])
			r = adversary.fit(y_disc, d_disc,
						      #callbacks=[TensorBoard(log_dir=args.tblog + '_D', write_graph=False)],
						      verbose=0)
			loss = r.history['loss'][0]
			mean_loss = mean_loss + loss
		return mean_loss / nsteps

	def train_generator(nsteps):
		mean_loss = 0.0
		for i in range(1,nsteps):
			batch_indeces = np.random.randint(0,O_train.shape[0],args.batch_size)
			o_in = O_train[batch_indeces,:,:,:]
			t_in = T_train[batch_indeces,:,:,:]
			y_in = Y_train[batch_indeces,:,:,:]
			r = generator.fit([o_in,t_in,y_in], [y_in, d_comb],
							 #callbacks=[TensorBoard(log_dir=args.tblog + '_G', write_graph=False)],
							 verbose=0)
			loss = r.history['loss'][0]
			mean_loss = mean_loss + loss
		return mean_loss / nsteps

	for step in range(1,args.nsteps):
		step = step + 1
		loss_D = train_discriminator(10)
		loss_G = train_generator(10)
		print "(%5d) D: %10.3f  G: %10.3f" % (step, loss_D, loss_G)

		if len(args.save) > 0:
			if args.verbosity > 0:
				print 'saving model weights to ', args.save
			generator.save_weights(args.save)

##########################
# Save Prediction

if len(args.save_pred) > 0:
	f = h5py.File(args.save_pred, 'w')
	pred = data=generator.predict([O_test, T_test, Y_test], batch_size=args.batch_size, verbose=args.verbosity)
	f.create_dataset("predict/Y_pred", data=pred[0], dtype="float32")
	f.create_dataset("predict/disc", data=pred[1], dtype="float32")
	pred = data=generator.predict([O_train, T_train, Y_train], batch_size=args.batch_size, verbose=args.verbosity)
	f.create_dataset("predict/Y_pred_train", data=pred[0], dtype="float32")
	f.create_dataset("predict/disc_train", data=pred[1], dtype="float32")
	f.close()
