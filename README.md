🦎 JOML: A minimalist `numpy`-backed Neural Network library
========================================================

#### 🚧 Work in progress! Don't try this (library) at home — *at least for now* !

Progress can be tracked by issues.


## Getting started

*JOML* API is quite similar to *Keras* [`Sequential Model` API](https://keras.io/getting-started/sequential-model-guide/).

```python
network = Network(input_size=42)

network.stack(Layer(size=38))
network.stack(Layer(size=38))

network.stack(Layer(size=16))
network.stack(Layer(size=16))

network.stack(Layer(size=8))

network.output(output_size=4)

network.train(x_train,y_train)
accuracy, network.test(x_test,y_test)
```

See [`main.py`](./main.py) for a simple example.
## Layers available

For now, there is just one type of `Layer` : Fully Connected Layer
By default, the `ReLu` is used as an activation function.

## Activation Functions

Two activation functions are available:
  - `ReLu`
  - `Sigmoid`

You can define your own `ActivationFunction` as well and use it for a `Layer`:

```python
class CustomActivation(ActivationFunction):

  def __init(self):
    value = lambda x: # what you want
    derivative = lambda x: # ∂(what you want) / ∂x  
    super().__init__(value, derivative)

network = Network(input_size=42)
# ...
network.stack(Layer(size=16,activation_function=CustomActivation()))
# ...
```

## Cost Functions

For now, only the `CrossEntropy` has been implemented.

You can define your own `CostFunction` as well and use it for a `Network`:

```python
class CustomCost(CostFunction):

  def __init(self):
    value = lambda x: # what you want
    derivative = lambda x: # ∂(what you want) / ∂x  
    super().__init__(value, derivative)

network = Network(input_size=42)
# ...
network.output(output_size=4, cost_function=CustomCost())
```

## Why does JOML mean?

JOML means "*JOML One More Layer*".


## License

[This project license](./LICENSE.md) is MIT.
