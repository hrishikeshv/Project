import os
import sys

import numpy as np

import theano
import theano.tensor as T
import argparse
import timeit
from adadelta import adadelta

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class PolyHidden(object):
    def __init__(self, rng, input, n_in, n_out, n_deg, W=None,activation=T.tanh):

        self.input = input
        self.deg = n_deg

        m = T.iscalar("m")
        d = T.iscalar("d")
        result, updates = theano.scan(fn=lambda L, m:L*(T.cast((T.arange(m)+1.0)/m, theano.config.floatX)),
                                        outputs_info=T.ones((m,)),
                                        non_sequences=m,
                                        n_steps=d)
        
        self.compute_index_matrix = theano.function(inputs=[m,d],outputs=T.concatenate([ T.ones((1,m)), result], axis=0))
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size = (n_deg + 1, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)


        self.W = W
        self.index_matrix = self.compute_index_matrix(n_in,self.deg).T

        lin_output = T.dot(input, T.dot(self.index_matrix, self.W))
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W]

def evaluate_lenet(args, learning_rate = 0.01, dataset='mnist.pkl.gz', hsize=[700,100], batch_size=256):

    rng = np.random.RandomState(23455)

    datasets = load_data(dataset)

    print(hsize)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    print('.. building model')

    layer0_input = x
    if args.l1 == "normal":
        layer0 = HiddenLayer(
                rng,
                input=layer0_input,
                n_in=28 * 28,
                n_out=hsize[0],
        )
    else:
        layer0 = PolyHidden(
                rng,
                input=layer0_input,
                n_in=28 * 28,
                n_out=hsize[0],
                n_deg=args.deg1
        )

    if args.l2 == "normal":
        layer1 = HiddenLayer(
            rng = rng,
            input = layer0.output,
            n_in = hsize[0],
            n_out = hsize[1],
            activation = T.tanh
        )
    else:
        layer1 = PolyHidden(
            rng = rng,
            input = layer0.output,
            n_in = hsize[0],
            n_out = hsize[1],
            n_deg = args.deg2,
            activation = T.tanh
        )
        

    layer2 = LogisticRegression(input=layer1.output, n_in=hsize[1], n_out=10)

    cost = layer2.negative_log_likelihood(y)

    test_model = theano.function(
        inputs=[index],
        outputs=layer2.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=layer2.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    params = layer2.params + layer1.params + layer0.params

    grads = T.grad(cost, params)

    updates = adadelta(params, grads)
#[
#        (param_i, param_i - learning_rate * grad_i)
#        for param_i,grad_i in zip(params,grads)
#    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < args.epoch) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l1", default= 'normal', help="use dense or polydense")
    parser.add_argument("--l2", default= 'normal', help="use dense or polydense")
    parser.add_argument("--deg1", default= 10, help="use dense or polydense", type=int)
    parser.add_argument("--deg2", default= 10, help="Input polynomial degree", type=int)
    parser.add_argument("--epoch",default= 100, help="Number of epochs", type=int) 
    parser.add_argument("--activ",default= "sigmoid", help="Number of hidden layers") 
    parser.add_argument("--reg", help="Regularization weight") 
    args = parser.parse_args(sys.argv[1:])
    evaluate_lenet(args)
