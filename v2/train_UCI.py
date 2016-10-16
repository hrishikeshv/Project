from keras.models import Sequential
from keras.layers import Dense, PolyDense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.initializations import normal
import sys
import argparse
import numpy as np
sys.path.insert(0, '../datasets/1d/')
np.random.seed(1234)

def train_model(xtrain, ytrain, xtest, ytest, args, input_dim):

	batch_size = min(len(xtrain), 32)
	nb_epoch =args.epoch 
	model = Sequential()

	W = normal((input_dim, args.nodes)).eval()
	coeff = np.polynomial.polynomial.polyfit(np.arange(input_dim) + 1.0, W, deg=args.deg)
	if args.mode == 'normal':
		print 'Using dense layer with {} nodes'.format(args.nodes)
		model.add(Dense(args.nodes, input_dim=input_dim, bias=False, weights=[W]))
	else:
		print 'Using polydense layer with {} nodes and degree {}'.format(args.nodes, args.deg)
		model.add(PolyDense(args.nodes, deg = args.deg, input_dim=input_dim, weights = [coeff]))

	model.add(Activation('relu'))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error',optimizer='adadelta')

	earlystopping = EarlyStopping(monitor = 'val_loss', patience=20, mode = 'min', verbose = 0)
	model.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[earlystopping], verbose=1, validation_data=(xtest, ytest))
	score = model.evaluate(xtest, ytest, verbose=0)
	print('Test score:', score)
	

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default= 'normal', help="use dense or polydense")
parser.add_argument("--deg", default= 10, help="Input polynomial degree", type=int)
parser.add_argument("--nodes", default= 10, help="Input polynomial degree", type=int)
parser.add_argument("--epoch",default=500, help="Number of epochs", type=int) 
parser.add_argument("f",default=1, help="Function to be trained", type=int) 
args = parser.parse_args(sys.argv[1:])

DIR_PATH = '../datasets/UCI/'
options = {1:'abalone', 2:'airfoil', 3:'concrete', 4:'housing'}
train_split = {1 : 500, 2 : 500, 3 : 500, 4 : 200}
data = np.load(DIR_PATH + options[args.f]+'.npz')
f = args.f
xtrain, ytrain = data['x'][:train_split[f]], data['y'][:train_split[f]]
xtest, ytest = data['x'][train_split[f]:], data['y'][train_split[f]:]

print xtrain.shape, xtest.shape
train_model(xtrain, ytrain, xtest, ytest, args, xtrain.shape[1])
