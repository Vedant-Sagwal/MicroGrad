A tiny Autograd engine. Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows.

[mlp.py](mlp.py) => Implements Multi layer Perceptron
[engine.py](engine.py) => implement Value class which is just like tensor but it is scalar only.
[test.py](test.py) => compares wit torch autograd
[Micrograd.ipynb](Micrograd.ipynb) => Contains all the code and demo and everything.
