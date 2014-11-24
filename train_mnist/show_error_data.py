import numpy as np
import pickle
import theano
from pylearn2.datasets import mnist
from pylearn2.gui import patch_viewer


def simulate(inputs, model):
    return model.fprop(theano.shared(inputs)).eval()

def show_errors(dataset, model, rows, cols, out=None):
    size = rows * cols
    errors = 0
    shape = dataset.view_shape()
    if shape[2] == 1:
        is_color = False
    else:
        is_color = True
    pv = patch_viewer.PatchViewer((rows, cols), shape[0:2], is_color=is_color)
    outputs = simulate(dataset.X, model)
    for x, output, label in zip(dataset.X, outputs, dataset.y):
        if np.argmax(output) != label:
            sample = dataset.get_topological_view(np.array([x]))
            sample = dataset.adjust_for_viewer(sample)
            pv.add_patch(sample[0], activation=0.0, rescale=False)
            print '{} {}'.format(np.argmax(output), label[0])
            errors += 1
            if errors >= size:
                break
    pv.show()

model = pickle.load(open('dae_mlp.pkl'))
test_data = mnist.MNIST(which_set='test')
show_errors(test_data, model, 20, 20)
