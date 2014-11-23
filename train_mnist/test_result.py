import numpy as np
import pickle
import theano
import pylearn2.datasets.mnist as mnist


def simulate(inputs, model):
    return model.fprop(theano.shared(inputs)).eval()

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

model = pickle.load(open('dae_mlp.pkl'))
test_data = mnist.MNIST(which_set='test')
print '%(correct)d / %(total)d' % score(test_data, model)
