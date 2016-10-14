from keras.models import Sequential
from keras.layers import Dense, PolyDense, Dropout, Activation, Flatten
from keras.utils import np_utils
import sys
import argparse
sys.path.insert(0, '../datasets/1d/')

from create_data import *

def nmse_score(pred, label):
	tot = np.sum(label**2)
	return 100.0*np.sum(np.power(pred - label, 2)) / tot

def train_model(xtrain, ytrain, xtest, ytest, args):

	batch_size = min(len(xtrain), 16)
	nb_epoch = 200
	model = Sequential()

	if args.mode == 'normal':
		model.add(Dense(args.nodes, input_dim=1))
	else:
		model.add(PolyDense(args.nodes, deg = args.deg, input_dim=1))

	model.add(Activation('relu'))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error',optimizer='adam')

	model.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(xtest, ytest))
	pred = model.predict(xtest, verbose=0).reshape(-1)
	print nmse_score(pred, ytest)
	

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default= 'normal', help="use dense or polydense")
parser.add_argument("--deg", default= 10, help="Input polynomial degree", type=int)
parser.add_argument("--nodes", default= 10, help="Input polynomial degree", type=int)

args = parser.parse_args(sys.argv[1:])

xtrain, ytrain = gen_b(1000)
xtest, ytest = gen_b(10000)


train_model(xtrain, ytrain, xtest, ytest, args)
