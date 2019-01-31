JOML: A minimalist `numpy`-baked Neural Network API 🦎 
========================================================

## Getting started

*JOML* API is quite similar to *Keras* [`Sequential Model` API](https://keras.io/getting-started/sequential-model-guide/).

It is for now available on [TestPypi](http://test.pypi.org/).
You can install it using this:

```bash
$ pip install --index-url https://test.pypi.org/project/ joml
```

```python
from joml.network import Network
from joml.layer import Layer, SoftMaxCrossEntropyOutputLayer
import numpy as np

# Loading/transforming data into np.ndarray
x_train, y_train, x_test, y_test = my_loader()
# Here those are arrays of respective shape :
# (14, num_examples) for x_train and x_test
# (4, num_examples) for y_train and y_test

# Defining your network
network = Network(input_size=14, name="My really first network")

network.stack(Layer(size=100))
network.stack(Layer(size=40))

network.output(SoftMaxCrossEntropyOutputLayer(size=4))

# Training
network.train(x_train,y_train)

# … wait (a bit) ⏳

# Profit ! 🚀
y_pred, y_hat, accuracy = network.test(x_test,y_test)
```

See [`examples`](./examples) for some examples.

## Features

The API is not definitive yet : *More to come soon !*

## Why does JOML mean?

JOML means "*JOML One More Layer*".

## License

[This project license](./LICENSE) is MIT.
