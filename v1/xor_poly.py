from create_xor import * 
from keras.models import Sequential
from keras.layers import Dense, PolyDense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

n = 10000
f = 2
def train_model(xtrain, ytrain, xtest, ytest):

	batch_size = 256
	nb_epoch = 200
	model = Sequential()
	model.add(PolyDense(4, input_dim=f))
	model.add(Activation('relu'))
	model.add(PolyDense(4))
	model.add(Activation('sigmoid'))
	model.add(Dense(2))
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
	
	model.fit(xtrain, ytrain, batch_size=batch_size, nb_epoch=nb_epoch,
			          verbose=1, validation_data=(xtest, ytest))
	score = model.evaluate(xtest, ytest, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])


if __name__ == "__main__" :
	
	trend = int(0.7 * n)
	feats,labels = xor_data(n, f)
#perm = np.random.permutation(len(feats))
#	feats = feats[perm]
#	labels = labels[perm]
	labels = np_utils.to_categorical(labels, 2)

	train_model(feats[:trend], labels[:trend], feats[trend:], labels[trend:])
	

