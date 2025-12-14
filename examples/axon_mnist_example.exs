#!/usr/bin/env elixir

# Example usage of the AxonMnist module
# This script demonstrates how to train and evaluate a model on MNIST data

# Make sure to download the MNIST dataset first:
# wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O ./tmp/train-images-idx3-ubyte.gz
# wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O ./tmp/train-labels-idx1-ubyte.gz
# gunzip ./tmp/train-images-idx3-ubyte.gz
# gunzip ./tmp/train-labels-idx1-ubyte.gz

# Simple training example
IO.puts("=== Simple MNIST Training with Axon ===")

# Create and train model
model = MachineLearning.AxonMnist.create_model()
params = MachineLearning.AxonMnist.init_params(model)

# Load training data
train_data = MachineLearning.Mnist.load("./tmp/train-images-idx3-ubyte", "./tmp/train-labels-idx1-ubyte", 32)

# Train the model
trained_params = MachineLearning.AxonMnist.train(model, params, train_data, epochs: 3, learning_rate: 0.001)

# Make a prediction on a single batch
{images, _labels} = Enum.at(train_data, 0)
predictions = MachineLearning.AxonMnist.predict(model, trained_params, images)
predicted_classes = MachineLearning.AxonMnist.predict_class(model, trained_params, images)

IO.puts("Predictions shape: #{inspect(Nx.shape(predictions))}")
IO.puts("First 5 predicted classes: #{inspect(Nx.to_list(predicted_classes) |> Enum.take(5))}")

# Complete workflow example (uncomment if you have test data)
result = MachineLearning.AxonMnist.train_and_evaluate(
  "./tmp/train-images-idx3-ubyte",
  "./tmp/train-labels-idx1-ubyte",
  "./tmp/t10k-images.idx3-ubyte",   # Download test data separately
  "./tmp/t10k-labels.idx1-ubyte",   # Download test data separately
  epochs: 5,
  batch_size: 32,
  learning_rate: 0.001
)

IO.puts("Training completed successfully!")
