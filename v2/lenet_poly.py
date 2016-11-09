'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
import argparse
import sys
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import PolyDense, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.initializations import normal

batch_size = 512 
nb_classes = 10
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


parser = argparse.ArgumentParser()
parser.add_argument("--l1", default= 'normal', help="use dense or polydense")
parser.add_argument("--l2", default= 'normal', help="use dense or polydense")
parser.add_argument("--deg1", default= 10, help="use dense or polydense", type=int)
parser.add_argument("--deg2", default= 10, help="Input polynomial degree", type=int)
parser.add_argument("--epoch",default= 100, help="Number of epochs", type=int) 
parser.add_argument("--activ",default= "sigmoid", help="Number of hidden layers") 
args = parser.parse_args(sys.argv[1:])
model = Sequential()

model.add(Convolution2D(6,5,5,border_mode="same",input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool),strides=(2,2)))
model.add(Convolution2D(16, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool),strides=(2,2)))

model.add(Flatten())

W = normal((model.output_shape[-1], 120)).eval()
if args.l1 == 'normal':	
	model.add(Dense(120, bias=False, weights=[W]))
else:
	coeff = np.polynomial.polynomial.polyfit(np.arange(model.output_shape[-1]) + 1.0, W, deg=args.deg1)
	model.add(PolyDense(120, deg = args.deg1,weights=[coeff])) 

model.add(Activation(args.activ)) 

W = normal((model.output_shape[-1], 84)).eval()
if args.l2 == 'normal':	
	model.add(Dense(84, bias=False, weights=[W]))
else:
	coeff = np.polynomial.polynomial.polyfit(np.arange(model.output_shape[-1]) + 1.0, W, deg=args.deg2)
	model.add(PolyDense(84, deg = args.deg2,weights=[coeff])) 
model.add(Activation(args.activ)) 
model.add(Dense(nb_classes)) 
model.add(Activation('softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
earlystopping = EarlyStopping(monitor = 'val_loss', patience=10, mode = 'min', verbose = 0)
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=args.epoch,callbacks=[earlystopping],
          verbose=0, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
with open('testscoreLenet5.txt','a') as f:
	f.write(str(score[0]) + ' ' + str(score[1])+'\n')
