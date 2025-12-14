defmodule MachineLearning.AxonMnist do
  @moduledoc """
  MNIST classifier using Axon library with a 3-layer neural network.

  This module provides functionality to:
  - Initialize a 3-layer neural network (784 -> 128 -> 10)
  - Train the model on MNIST dataset
  - Evaluate the model performance
  """

  @doc """
  Creates and initializes a 3-layer neural network model for MNIST classification.

  The model architecture:
  - Input layer: 784 neurons (28x28 flattened images)
  - Hidden layer: 128 neurons with ReLU activation
  - Output layer: 10 neurons (one for each digit) with softmax activation

  ## Examples

      iex> model = MachineLearning.AxonMnist.create_model()
  """
  @spec create_model() :: Axon.t()
  def create_model do
    Axon.input("input", shape: {nil, 784})
    |> Axon.dense(128, activation: :sigmoid, name: "hidden")
    |> Axon.dense(10, activation: :softmax, name: "output")
  end

  @doc """
  Initializes the model parameters.

  ## Parameters

  - `model`: The Axon model to initialize
  - `key`: Random key for parameter initialization (optional)

  ## Examples

      iex> model = MachineLearning.AxonMnist.create_model()
      iex> params = MachineLearning.AxonMnist.init_params(model)
  """
  @spec init_params(Axon.t(), Nx.Tensor.t() | nil) :: map()
  def init_params(model, _key \\ nil) do
    # Use a simple approach - just create a sample input and let Axon handle initialization
    # sample_input = %{"input" => Nx.iota({1, 784}, type: :f32)}
    {init_fun, _} = Axon.build(model)

    init_fun.(Nx.template({1, 784}, {:f, 32}), Axon.ModelState.new(%{}))
  end

  @doc """
  Trains the model on the MNIST dataset.

  ## Parameters

  - `model`: The Axon model to train
  - `params`: Initial model parameters
  - `train_data`: Training dataset (stream of {images, labels} batches)
  - `opts`: Training options (optional)
    - `:epochs` - Number of training epochs (default: 10)
    - `:learning_rate` - Learning rate for optimizer (default: 0.001)
    - `:optimizer` - Optimizer to use (default: :adam)

  ## Examples

      iex> model = MachineLearning.AxonMnist.create_model()
      iex> params = MachineLearning.AxonMnist.init_params(model)
      iex> train_data = MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte", 32)
      iex> trained_params = MachineLearning.AxonMnist.train(model, params, train_data, epochs: 5)
  """
  @spec train(Axon.t(), map(), Enumerable.t(), keyword()) :: map()
  def train(model, params, train_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    optimizer = Keyword.get(opts, :optimizer, :adam)

    # Convert data format for Axon
    train_data =
      train_data
      |> Stream.map(fn {images, labels} ->
        {%{"input" => images}, labels}
      end)

    # Create loss function
    loss_fn = fn y_true, y_pred ->
      Axon.Losses.categorical_cross_entropy(y_true, y_pred, reduction: :mean)
    end

    # Create optimizer
    optimizer_fn =
      case optimizer do
        :adam -> Polaris.Optimizers.adam(learning_rate: learning_rate)
        :sgd -> Polaris.Optimizers.sgd(learning_rate: learning_rate)
        _ -> Polaris.Optimizers.adam(learning_rate: learning_rate)
      end

    # Train the model
    model
    |> Axon.Loop.trainer(loss_fn, optimizer_fn)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(train_data, params, epochs: epochs, compiler: EXLA)
  end

  @doc """
  Evaluates the model on test data.

  ## Parameters

  - `model`: The Axon model to evaluate
  - `params`: Trained model parameters
  - `test_data`: Test dataset (stream of {images, labels} batches)
  - `opts`: Evaluation options (optional)
    - `:show_failures` - Show misclassified images (default: false)
    - `:max_failures` - Maximum number of failures to show (default: 5)

  ## Examples

      iex> test_data = MachineLearning.Mnist.load("./tmp/test-images-idx3-ubyte", "./tmp/test-labels-idx1-ubyte", 32)
      iex> results = MachineLearning.AxonMnist.evaluate(model, trained_params, test_data, show_failures: true)
  """
  @spec evaluate(Axon.t(), map(), Enumerable.t(), keyword()) :: map()
  def evaluate(model, params, test_data, opts \\ []) do
    show_failures = Keyword.get(opts, :show_failures, false)
    max_failures = Keyword.get(opts, :max_failures, 5)

    # Convert data format for Axon
    axon_test_data =
      test_data
      |> Stream.map(fn {images, labels} ->
        {%{"input" => images}, labels}
      end)

    # Evaluate the model
    results =
      model
      |> Axon.Loop.evaluator()
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.run(axon_test_data, params, compiler: EXLA)

    # Show failure analysis if requested
    if show_failures do
      IO.puts("\n=== Failure Analysis ===")
      inspect_failures(model, params, test_data, max_failures)
    end

    results
  end

  @doc """
  Makes predictions on new data.

  ## Parameters

  - `model`: The Axon model
  - `params`: Trained model parameters
  - `input`: Input tensor of shape {batch_size, 784}

  ## Examples

      iex> predictions = MachineLearning.AxonMnist.predict(model, trained_params, images)
  """
  @spec predict(Axon.t(), map(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def predict(model, params, input) do
    Axon.predict(model, params, %{"input" => input}, compiler: EXLA)
  end

  @doc """
  Predicts the class (digit) for given input images.

  ## Parameters

  - `model`: The Axon model
  - `params`: Trained model parameters
  - `input`: Input tensor of shape {batch_size, 784}

  Returns the predicted class indices.

  ## Examples

      iex> predicted_classes = MachineLearning.AxonMnist.predict_class(model, trained_params, images)
  """
  @spec predict_class(Axon.t(), map(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def predict_class(model, params, input) do
    model
    |> predict(params, input)
    |> Nx.argmax(axis: -1)
  end

  @doc """
  Complete workflow: create, train, and evaluate a model on MNIST data.

  ## Parameters

  - `train_images_path`: Path to training images file
  - `train_labels_path`: Path to training labels file
  - `test_images_path`: Path to test images file (optional)
  - `test_labels_path`: Path to test labels file (optional)
  - `opts`: Training options

  ## Examples

      iex> result = MachineLearning.AxonMnist.train_and_evaluate(
      ...>   "./tmp/train-images.idx3-ubyte",
      ...>   "./tmp/train-labels.idx1-ubyte",
      ...>   epochs: 10,
      ...>   batch_size: 32
      ...> )
  """
  @spec train_and_evaluate(String.t(), String.t(), String.t() | nil, String.t() | nil, keyword()) ::
          map()
  def train_and_evaluate(
        train_images_path,
        train_labels_path,
        test_images_path \\ nil,
        test_labels_path \\ nil,
        opts \\ []
      ) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    epochs = Keyword.get(opts, :epochs, 10)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)

    IO.puts("Creating model...")
    model = create_model()
    params = init_params(model)

    IO.puts("Loading training data...")
    train_data = MachineLearning.Mnist.load(train_images_path, train_labels_path, batch_size)

    IO.puts("Starting training...")
    start_time = System.monotonic_time()

    trained_params =
      train(model, params, train_data, epochs: epochs, learning_rate: learning_rate)

    end_time = System.monotonic_time()
    training_time = System.convert_time_unit(end_time - start_time, :native, :second)

    IO.puts("Training completed in #{training_time} seconds")

    result = %{
      model: model,
      params: trained_params,
      training_time: training_time
    }

    # Evaluate on test data if provided
    if test_images_path && test_labels_path do
      IO.puts("Loading test data...")
      test_data = MachineLearning.Mnist.load(test_images_path, test_labels_path, batch_size)

      IO.puts("Evaluating model...")

      test_results =
        evaluate(model, trained_params, test_data, show_failures: true, max_failures: 3)

      IO.puts("Test Results:")
      IO.inspect(test_results)

      Map.put(result, :test_results, test_results)
    else
      result
    end
  end

  @doc """
  Displays a random prediction from the test set with visualization.

  ## Parameters

  - `model`: The trained Axon model
  - `params`: Trained model parameters
  - `test_images_path`: Path to test images file
  - `test_labels_path`: Path to test labels file
  - `batch_size`: Batch size for loading data (default: 1)

  ## Examples

      iex> MachineLearning.AxonMnist.show_prediction(model, trained_params, "./tmp/test-images-idx3-ubyte", "./tmp/test-labels-idx1-ubyte")
  """
  @spec show_prediction(Axon.t(), map(), String.t(), String.t(), integer()) :: :ok
  def show_prediction(model, params, test_images_path, test_labels_path, batch_size \\ 32) do
    test_data = MachineLearning.Mnist.load(test_images_path, test_labels_path, batch_size)

    {images, labels} = Enum.random(test_data)
    {batch_size, _} = Nx.shape(images)
    index = Enum.random(0..(batch_size - 1))

    predicted_probs = predict(model, params, images)

    predicted_class =
      predicted_probs
      |> Nx.argmax(axis: -1)
      |> Nx.to_list()
      |> Enum.at(index)

    confidence =
      predicted_probs
      |> Nx.to_list()
      |> Enum.at(index)
      |> Enum.at(predicted_class)
      |> Float.round(4)

    {display, expected} = MachineLearning.Mnist.inspect(images, labels, index)

    IO.puts(display)
    IO.puts("Expected: #{expected}")
    IO.puts("Predicted: #{predicted_class}")
    IO.puts("Confidence: #{confidence * 100}%")

    :ok
  end

  @doc """
  Inspects misclassified images from the test data.

  ## Parameters

  - `model`: The trained Axon model
  - `params`: Trained model parameters
  - `test_data`: Test dataset (stream of {images, labels} batches)
  - `max_failures`: Maximum number of failures to show (default: 5)

  ## Examples

      iex> test_data = MachineLearning.Mnist.load("./tmp/test-images-idx3-ubyte", "./tmp/test-labels-idx1-ubyte", 32)
      iex> MachineLearning.AxonMnist.inspect_failures(model, trained_params, test_data, 10)
  """
  @spec inspect_failures(Axon.t(), map(), Enumerable.t(), integer()) :: :ok
  def inspect_failures(model, params, test_data, max_failures \\ 5) do
    failures_found = 0

    test_data
    |> Stream.take_while(fn _ -> failures_found < max_failures end)
    |> Enum.reduce_while(0, fn {images, labels}, acc ->
      if acc >= max_failures do
        {:halt, acc}
      else
        new_failures = find_and_show_failures(model, params, images, labels, max_failures - acc)
        {:cont, acc + new_failures}
      end
    end)

    :ok
  end

  # Private function to find and display failures in a batch
  defp find_and_show_failures(model, params, images, labels, remaining_slots) do
    predictions = predict(model, params, images)
    predicted_classes = Nx.argmax(predictions, axis: -1) |> Nx.to_list()

    actual_labels =
      labels
      |> Nx.argmax(axis: -1)
      |> Nx.to_list()

    predicted_probs = Nx.to_list(predictions)

    images
    |> Nx.to_list()
    |> Enum.with_index()
    |> Enum.reduce_while(0, fn {_image, index}, failures_shown ->
      if failures_shown >= remaining_slots do
        {:halt, failures_shown}
      else
        predicted = Enum.at(predicted_classes, index)
        actual = Enum.at(actual_labels, index)

        if predicted != actual do
          show_failure(images, labels, predicted_probs, index, predicted, actual)
          {:cont, failures_shown + 1}
        else
          {:cont, failures_shown}
        end
      end
    end)
  end

  # Private function to display a single misclassification
  defp show_failure(images, labels, predicted_probs, index, predicted, actual) do
    confidence =
      predicted_probs
      |> Enum.at(index)
      |> Enum.at(predicted)
      |> Float.round(4)

    {display, _} = MachineLearning.Mnist.inspect(images, labels, index)

    IO.puts("\n#{IO.ANSI.red()}=== MISCLASSIFICATION ===#{IO.ANSI.reset()}")
    IO.puts(display)
    IO.puts("#{IO.ANSI.green()}Expected: #{actual}#{IO.ANSI.reset()}")
    IO.puts("#{IO.ANSI.red()}Predicted: #{predicted}#{IO.ANSI.reset()}")
    IO.puts("#{IO.ANSI.yellow()}Confidence: #{confidence * 100}%#{IO.ANSI.reset()}")
    IO.puts("#{String.duplicate("=", 30)}")
  end
end
