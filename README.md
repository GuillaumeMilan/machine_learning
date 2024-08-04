# MachineLearning

Implementation in Elixir of the deep-learning course from
[Youtube](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1)


## Chapter 1: What is a neural network?

Executing a model not trained on a random image:
```elixir
random_image = MachineLearning.Activation.generate_random(784)
model = MachineLearning.Model.init([784, 16, 16, 10])
MachineLearning.Model.execute(model, random_image)
```

- TODO: Read images from the [MNIST](https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data)
