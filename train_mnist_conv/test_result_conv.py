import numpy as np
import pickle
import theano
import pylearn2.datasets.mnist as mnist
from pylearn2.space import VectorSpace

def simulate(inputs, model):
    space = VectorSpace(inputs.shape[1])
    X = space.make_theano_batch()
    Y = model.fprop(space.format_as(X, model.get_input_space()))
    f = theano.function([X], Y)
    return f(inputs)

def countCorrectResults(outputs, labels):
    correct = 0;
    for output, label in zip(outputs, labels):
        if np.argmax(output) == label:
            correct += 1
    return correct
 
def score(dataset, model):
    outputs = simulate(dataset.X, model)
    correct = countCorrectResults(outputs, dataset.y)

    return {
        'correct': correct,
        'total': len(dataset.X)
    }

model = pickle.load(open('convolutional_network_best.pkl'))
test_data = mnist.MNIST(which_set='test')
print '%(correct)d / %(total)d' % score(test_data, model)
