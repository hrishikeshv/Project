from keras.models import Sequential
from keras.layers import Dense, PolyDense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.initializations import normal
import sys
import argparse
sys.path.insert(0, '../datasets/1d/')

from create_data import *

def nmse_score(pred, label):
	tot = np.sum(label**2)
	return 100.0*np.sum(np.power(pred - label, 2)) / tot

def train_model(xtrain, ytrain, xtest, ytest, args):

	batch_size = min(len(xtrain), 16)
	nb_epoch = 500
	model = Sequential()

	W = normal((1,args.nodes)).eval()
	coeff = np.polynomial.polynomial.polyfit(np.arange(1) + 1.0, W, deg=args.deg)
	if args.mode == 'normal':
		print 'Using dense layer with {} nodes'.format(args.nodes)
		model.add(Dense(args.nodes, input_dim=1, bias=False, weights=[W]))
	else:
		print 'Using polydense layer with {} nodes and degree {}'.format(args.nodes, args.deg)
		model.add(PolyDense(args.nodes, deg = args.deg, input_dim=1, weights = [coeff]))

	model.add(Activation('relu'))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error',optimizer='adadelta')

	earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min', verbose = 0)
	model.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[earlystopping], verbose=1, validation_data=(xtest, ytest))
	score = model.evaluate(xtest, ytest, verbose=0)
	print('Test score:', score)
	

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default= 'normal', help="use dense or polydense")
parser.add_argument("--deg", default= 10, help="Input polynomial degree", type=int)
parser.add_argument("--nodes", default= 10, help="Input polynomial degree", type=int)
parser.add_argument("f",default=1, help="Function to be trained", type=int) 
args = parser.parse_args(sys.argv[1:])

options = {1:gen_a, 2:gen_b, 3:gen_c, 4:gen_d}
f = options[args.f]
xtrain, ytrain = f(1000)
xtest, ytest = f(10000)

train_model(xtrain, ytrain, xtest, ytest, args)
