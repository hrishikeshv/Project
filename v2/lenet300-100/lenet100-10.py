'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
import argparse
import sys
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, PolyDense, Dropout, Activation 
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.initializations import normal
from keras.regularizers import l2, l1


batch_size = 128
nb_classes = 10
nb_epoch = 20
input_dim = 784
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
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
parser.add_argument("--reg", help="Regularization weight") 
args = parser.parse_args(sys.argv[1:])

reg = None
if args.reg:
	if args.reg[:2] == "l1":
		reg = l1(float(args.reg[2:]))
	elif args.reg[:2] == "l2":
		reg = l2(float(args.reg[2:]))
model = Sequential()
W = normal((input_dim, 10)).eval()
if args.l1 == 'normal':	
	model.add(Dense(10, input_dim=input_dim, bias=False, weights=[W]))
else:
	coeff = np.polynomial.polynomial.polyfit(np.arange(input_dim) + 1.0, W, deg=args.deg1)
	model.add(PolyDense(100, input_dim=input_dim, deg = args.deg1, weights=[coeff], W_regularizer = reg)) 

model.add(Activation(args.activ)) 

W = normal((model.output_shape[-1], 10)).eval()
if args.l2 == 'normal':
	model.add(Dense(10, bias=False, weights=[W], trainable=False))
else:
	coeff = np.polynomial.polynomial.polyfit(np.arange(model.output_shape[-1]) + 1.0, W, deg=args.deg2)
	model.add(PolyDense(10, deg = args.deg2, weights=[coeff])) 
model.add(Activation(args.activ)) 
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta',
			  metrics=['accuracy'])

print(model.summary())
earlystopping = EarlyStopping(monitor = 'val_loss', patience=10, mode = 'min', verbose = 0)
model.fit(X_train, Y_train,
					batch_size=batch_size, nb_epoch=args.epoch, callbacks=[earlystopping],
					verbose=1, validation_split=0.1)
score = model.evaluate(X_test, Y_test, verbose=0)
print(score[1])
#with open('testscoreLenet300-100_reg.txt','a') as f:
#	f.write(str(score[0]) + ' ' + str(score[1])+'\n')
