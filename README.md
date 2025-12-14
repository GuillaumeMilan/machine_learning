# MachineLearning

Implementation in Elixir of the deep-learning course from
[Youtube](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1)


## Chapter 1: What is a neural network?

Executing a model not trained on a random image:
```elixir
random_image = MachineLearning.Activation.generate_random(784)
model = MachineLearning.Network.init([784, 16, 16, 10])
MachineLearning.Network.execute(model, random_image)
```

## Chapter 2: Gradient descent

[MNIST Dataset](https://github.com/golbin/TensorFlow-MNIST/tree/master/mnist/data)
Train a model with MNIST dataset:
```elixir
model = MachineLearning.Network.init([784, 128, 10], 0.01)
set = MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte", 30)
model = set
    |> Enum.reduce(model, fn {image, label}, model -> MachineLearning.Network.train(model, image, label) end)
```

Detailed script in: `./scripts/mnist.exs`
