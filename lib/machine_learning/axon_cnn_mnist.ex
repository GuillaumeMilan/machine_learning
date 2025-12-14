defmodule MachineLearning.AxonCnnMnist do
  @moduledoc """
  MNIST classifier using Axon library with a Convolutional Neural Network.

  This module provides functionality to:
  - Initialize a CNN model for MNIST classification
  - Train the model on MNIST dataset
  - Evaluate the model performance

  The CNN architecture typically achieves better accuracy than fully-connected networks
  on image classification tasks by preserving spatial relationships in the data.

        # Train a model first
        result = MachineLearning.AxonCnnMnist.train_and_evaluate(
          "./tmp/train-images.idx3-ubyte",
          "./tmp/train-labels.idx1-ubyte",
          "./tmp/t10k-images.idx3-ubyte",
          "./tmp/t10k-labels.idx1-ubyte",
          epochs: 5
        )

        # Then analyze predictions with feature maps
        MachineLearning.AxonCnnMnist.analyze_prediction(
          result.model,
          result.params,
          "./tmp/t10k-images.idx3-ubyte",
          "./tmp/t10k-labels.idx1-ubyte",
          show_feature_maps: true,
          layers: ["conv1", "conv2"]
        )
  """

  @doc """
  Creates and initializes a Convolutional Neural Network model for MNIST classification.

  The model architecture:
  - Input: 28x28x1 images (reshaped from 784 flattened)
  - Conv2D: 32 filters, 3x3 kernel, ReLU activation
  - MaxPool2D: 2x2 pool size
  - Conv2D: 64 filters, 3x3 kernel, ReLU activation
  - MaxPool2D: 2x2 pool size
  - Flatten
  - Dense: 128 neurons, ReLU activation
  - Dropout: 0.5 rate (for regularization)
  - Dense: 10 neurons, softmax activation (output)

  ## Examples

      iex> model = MachineLearning.AxonCnnMnist.create_model()
      iex> result = MachineLearning.AxonCnnMnist.train_and_evaluate(
            "./tmp/train-images.idx3-ubyte",
            "./tmp/train-labels.idx1-ubyte",
            "./tmp/t10k-images.idx3-ubyte",
            "./tmp/t10k-labels.idx1-ubyte",
            epochs: 5,
            model_type: :simple
          )
  """
  @spec create_model() :: Axon.t()
  def create_model do
    Axon.input("input", shape: {nil, 28, 28, 1})
    |> Axon.conv(32, kernel_size: 3, padding: :same, activation: :relu, name: "conv1")
    |> Axon.max_pool(kernel_size: 2, name: "pool1")
    |> Axon.conv(64, kernel_size: 3, padding: :same, activation: :relu, name: "conv2")
    |> Axon.max_pool(kernel_size: 2, name: "pool2")
    |> Axon.flatten(name: "flatten")
    |> Axon.dense(128, activation: :relu, name: "dense1")
    |> Axon.dropout(rate: 0.5, name: "dropout")
    |> Axon.dense(10, activation: :softmax, name: "output")
  end

  @doc """
  Creates a simpler CNN model for faster training and experimentation.

  The model architecture:
  - Input: 28x28x1 images
  - Conv2D: 16 filters, 5x5 kernel, ReLU activation
  - MaxPool2D: 2x2 pool size
  - Conv2D: 32 filters, 5x5 kernel, ReLU activation
  - MaxPool2D: 2x2 pool size
  - Flatten
  - Dense: 64 neurons, ReLU activation
  - Dense: 10 neurons, softmax activation

  ## Examples

      iex> model = MachineLearning.AxonCnnMnist.create_simple_model()
  """
  @spec create_simple_model() :: Axon.t()
  def create_simple_model do
    Axon.input("input", shape: {nil, 28, 28, 1})
    |> Axon.conv(16, kernel_size: 3, padding: :same, activation: :relu, name: "conv1")
    |> Axon.max_pool(kernel_size: 2, name: "pool1")
    |> Axon.conv(32, kernel_size: 3, padding: :same, activation: :relu, name: "conv2")
    |> Axon.max_pool(kernel_size: 2, name: "pool2")
    |> Axon.flatten(name: "flatten")
    |> Axon.dense(64, activation: :relu, name: "dense1")
    |> Axon.dense(10, activation: :softmax, name: "output")
  end

  @doc """
  Initializes the CNN model parameters.

  ## Parameters

  - `model`: The Axon CNN model to initialize
  - `key`: Random key for parameter initialization (optional)

  ## Examples

      iex> model = MachineLearning.AxonCnnMnist.create_model()
      iex> params = MachineLearning.AxonCnnMnist.init_params(model)
  """
  @spec init_params(Axon.t(), Nx.Tensor.t() | nil) :: map()
  def init_params(model, _key \\ nil) do
    {init_fun, _} = Axon.build(model)

    # Create a proper template and state for initialization
    template = Nx.template({1, 28, 28, 1}, :f32)
    init_fun.(template, Axon.ModelState.new(%{}))
  end

  @doc """
  Preprocesses MNIST data to the correct format for CNN training.

  Reshapes the flattened 784-element vectors into 28x28x1 images.

  ## Parameters

  - `images`: Tensor of shape {batch_size, 784}
  - `labels`: Tensor of shape {batch_size, 10} (one-hot encoded)

  Returns: {reshaped_images, labels} where images have shape {batch_size, 1, 28, 28}

  ## Examples

      iex> {cnn_images, labels} = MachineLearning.AxonCnnMnist.preprocess_data(images, labels)
  """
  @spec preprocess_data(Nx.Tensor.t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def preprocess_data(images, labels) do
    {batch_size, _} = Nx.shape(images)
    # Reshape from {batch_size, 784} to {batch_size, 28, 28, 1} (channel-last format)
    reshaped_images = Nx.reshape(images, {batch_size, 28, 28, 1})
    {reshaped_images, labels}
  end

  @doc """
  Trains the CNN model on the MNIST dataset.

  ## Parameters

  - `model`: The Axon CNN model to train
  - `params`: Initial model parameters
  - `train_data`: Training dataset (stream of {images, labels} batches)
  - `opts`: Training options (optional)
    - `:epochs` - Number of training epochs (default: 10)
    - `:learning_rate` - Learning rate for optimizer (default: 0.001)
    - `:optimizer` - Optimizer to use (default: :adam)

  ## Examples

      iex> model = MachineLearning.AxonCnnMnist.create_model()
      iex> params = MachineLearning.AxonCnnMnist.init_params(model)
      iex> train_data = MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte", 32)
      iex> trained_params = MachineLearning.AxonCnnMnist.train(model, params, train_data, epochs: 5)
  """
  @spec train(Axon.t(), map(), Enumerable.t(), keyword()) :: map()
  def train(model, params, train_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    optimizer = Keyword.get(opts, :optimizer, :adam)

    # Convert and preprocess data format for CNN
    train_data =
      train_data
      |> Stream.map(fn {images, labels} ->
        {cnn_images, labels} = preprocess_data(images, labels)
        {%{"input" => cnn_images}, labels}
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
        :adamw -> Polaris.Optimizers.adamw(learning_rate: learning_rate)
        _ -> Polaris.Optimizers.adam(learning_rate: learning_rate)
      end

    # Train the model
    model
    |> Axon.Loop.trainer(loss_fn, optimizer_fn)
    |> Axon.Loop.metric(:accuracy)
    |> Axon.Loop.run(train_data, params, epochs: epochs, compiler: EXLA)
  end

  @doc """
  Evaluates the CNN model on test data.

  ## Parameters

  - `model`: The Axon CNN model to evaluate
  - `params`: Trained model parameters
  - `test_data`: Test dataset (stream of {images, labels} batches)
  - `opts`: Evaluation options (optional)
    - `:show_failures` - Show misclassified images (default: false)
    - `:max_failures` - Maximum number of failures to show (default: 5)

  ## Examples

      iex> test_data = MachineLearning.Mnist.load("./tmp/test-images-idx3-ubyte", "./tmp/test-labels-idx1-ubyte", 32)
      iex> results = MachineLearning.AxonCnnMnist.evaluate(model, trained_params, test_data, show_failures: true)
  """
  @spec evaluate(Axon.t(), map(), Enumerable.t(), keyword()) :: map()
  def evaluate(model, params, test_data, opts \\ []) do
    show_failures = Keyword.get(opts, :show_failures, false)
    max_failures = Keyword.get(opts, :max_failures, 5)

    # Convert and preprocess data format for CNN
    axon_test_data =
      test_data
      |> Stream.map(fn {images, labels} ->
        {cnn_images, labels} = preprocess_data(images, labels)
        {%{"input" => cnn_images}, labels}
      end)

    # Evaluate the model
    results =
      model
      |> Axon.Loop.evaluator()
      |> Axon.Loop.metric(:accuracy)
      |> Axon.Loop.run(axon_test_data, params, compiler: EXLA)

    # Show failure analysis if requested
    if show_failures do
      IO.puts("\n=== CNN Failure Analysis ===")
      inspect_failures(model, params, test_data, max_failures)
    end

    results
  end

  @doc """
  Makes predictions on new data using the CNN model.

  ## Parameters

  - `model`: The Axon CNN model
  - `params`: Trained model parameters
  - `input`: Input tensor of shape {batch_size, 784} (will be reshaped to CNN format)

  ## Examples

      iex> predictions = MachineLearning.AxonCnnMnist.predict(model, trained_params, images)
  """
  @spec predict(Axon.t(), map(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def predict(model, params, input) do
    # Reshape input if needed (from 784 to 28x28x1)
    {batch_size, features} = Nx.shape(input)

    cnn_input =
      case features do
        784 -> Nx.reshape(input, {batch_size, 28, 28, 1})
        # Assume already in correct CNN format
        _ -> input
      end

    Axon.predict(model, params, %{"input" => cnn_input}, compiler: EXLA)
  end

  @doc """
  Predicts the class (digit) for given input images using the CNN model.

  ## Parameters

  - `model`: The Axon CNN model
  - `params`: Trained model parameters
  - `input`: Input tensor of shape {batch_size, 784} or {batch_size, 1, 28, 28}

  Returns the predicted class indices.

  ## Examples

      iex> predicted_classes = MachineLearning.AxonCnnMnist.predict_class(model, trained_params, images)
  """
  @spec predict_class(Axon.t(), map(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def predict_class(model, params, input) do
    model
    |> predict(params, input)
    |> Nx.argmax(axis: -1)
  end

  @doc """
  Complete workflow: create, train, and evaluate a CNN model on MNIST data.

  ## Parameters

  - `train_images_path`: Path to training images file
  - `train_labels_path`: Path to training labels file
  - `test_images_path`: Path to test images file (optional)
  - `test_labels_path`: Path to test labels file (optional)
  - `opts`: Training options
    - `:model_type` - :simple or :full (default: :full)
    - `:epochs` - Number of training epochs
    - `:batch_size` - Batch size for training
    - `:learning_rate` - Learning rate

  ## Examples

      iex> result = MachineLearning.AxonCnnMnist.train_and_evaluate(
      ...>   "./tmp/train-images.idx3-ubyte",
      ...>   "./tmp/train-labels.idx1-ubyte",
      ...>   epochs: 10,
      ...>   batch_size: 32,
      ...>   model_type: :simple
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
    model_type = Keyword.get(opts, :model_type, :full)

    IO.puts("Creating CNN model (#{model_type})...")

    model =
      case model_type do
        :simple -> create_simple_model()
        _ -> create_model()
      end

    params = init_params(model)

    IO.puts("Loading training data...")
    train_data = MachineLearning.Mnist.load(train_images_path, train_labels_path, batch_size)

    IO.puts("Starting CNN training...")
    start_time = System.monotonic_time()

    trained_params =
      train(model, params, train_data, epochs: epochs, learning_rate: learning_rate)

    end_time = System.monotonic_time()
    training_time = System.convert_time_unit(end_time - start_time, :native, :second)

    IO.puts("CNN training completed in #{training_time} seconds")

    result = %{
      model: model,
      params: trained_params,
      training_time: training_time,
      model_type: model_type
    }

    # Evaluate on test data if provided
    if test_images_path && test_labels_path do
      IO.puts("Loading test data...")
      test_data = MachineLearning.Mnist.load(test_images_path, test_labels_path, batch_size)

      IO.puts("Evaluating CNN model...")

      test_results =
        evaluate(model, trained_params, test_data, show_failures: true, max_failures: 3)

      IO.puts("CNN Test Results:")
      IO.inspect(test_results)

      Map.put(result, :test_results, test_results)
    else
      result
    end
  end

  @doc """
  Displays a random prediction from the test set with CNN visualization.

  ## Parameters

  - `model`: The trained Axon CNN model
  - `params`: Trained model parameters
  - `test_images_path`: Path to test images file
  - `test_labels_path`: Path to test labels file
  - `batch_size`: Batch size for loading data (default: 32)

  ## Examples

      iex> MachineLearning.AxonCnnMnist.show_prediction(model, trained_params, "./tmp/test-images-idx3-ubyte", "./tmp/test-labels-idx1-ubyte")
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
    IO.puts("Model Type: Convolutional Neural Network")

    :ok
  end

  @doc """
  Visualizes the feature maps (activations) from convolutional layers for a given input image.

  This function extracts and displays the intermediate representations learned by the CNN,
  showing how the network transforms the input image through different convolutional layers.

  ## Parameters

  - `model`: The trained Axon CNN model
  - `params`: Trained model parameters
  - `input_image`: Single image tensor of shape {1, 28, 28, 1} or {1, 784}
  - `opts`: Visualization options
    - `:layers` - List of layer names to visualize (default: ["conv1", "conv2"])
    - `:max_filters` - Maximum number of filters to show per layer (default: 8)
    - `:show_original` - Whether to show the original image (default: true)

  ## Examples

      iex> # Get a single image from test data
      iex> test_data = MachineLearning.Mnist.load("./tmp/t10k-images.idx3-ubyte", "./tmp/t10k-labels.idx1-ubyte", 1)
      iex> {images, labels} = Enum.take(test_data, 1) |> List.first()
      iex> single_image = images |> Nx.slice_along_axis(0, 1, axis: 0)
      iex> MachineLearning.AxonCnnMnist.inspect_feature_maps(model, trained_params, single_image)
  """
  @spec inspect_feature_maps(Axon.t(), map(), Nx.Tensor.t(), keyword()) :: :ok
  def inspect_feature_maps(model, params, input_image, opts \\ []) do
    layers_to_show = Keyword.get(opts, :layers, ["conv1", "conv2"])
    max_filters = Keyword.get(opts, :max_filters, 8)
    show_original = Keyword.get(opts, :show_original, true)

    # Ensure input is in correct format - handle different input shapes
    formatted_input = case Nx.shape(input_image) do
      {batch_size, 784} ->
        # Flattened format, reshape to CNN format
        Nx.reshape(input_image, {batch_size, 28, 28, 1})
      {_, 28, 28, 1} ->
        # Already in CNN format
        input_image
      _ ->
        # Fallback: try to ensure it's in the right format
        input_image
    end

    if show_original do
      IO.puts("\n#{IO.ANSI.cyan()}=== Original Input Image ===#{IO.ANSI.reset()}")
      display_image_tensor(formatted_input)
    end

    # Extract feature maps from specified layers
    Enum.each(layers_to_show, fn layer_name ->
      case extract_layer_activations(model, params, formatted_input, layer_name) do
        {:ok, activations} ->
          display_feature_maps(activations, layer_name, max_filters)
        {:error, reason} ->
          IO.puts("#{IO.ANSI.red()}Failed to extract activations for #{layer_name}: #{reason}#{IO.ANSI.reset()}")
      end
    end)

    :ok
  end

  @doc """
  Extracts activations from a specific layer in the CNN model.

  ## Parameters

  - `model`: The Axon CNN model
  - `params`: Trained model parameters
  - `input`: Input tensor of shape {1, 28, 28, 1}
  - `layer_name`: Name of the layer to extract activations from

  Returns `{:ok, activations}` on success or `{:error, reason}` on failure.
  """
  @spec extract_layer_activations(Axon.t(), map(), Nx.Tensor.t(), String.t()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def extract_layer_activations(_model, params, input, layer_name) do
    try do
      # For now, we'll create truncated models to extract layer activations
      # Note: This is a simplified approach - full intermediate extraction
      # would require modifying the model architecture
      case layer_name do
        "conv1" ->
          # For demonstration, we'll create a truncated model up to conv1
          truncated_model = Axon.input("input", shape: {nil, 28, 28, 1})
                           |> Axon.conv(32, kernel_size: 3, padding: :same, activation: :relu, name: "conv1")

          {_init_fun, truncated_predict_fun} = Axon.build(truncated_model, mode: :inference)
          activations = truncated_predict_fun.(params, %{"input" => input})
          {:ok, activations}

        "conv2" ->
          # Create model up to conv2
          truncated_model = Axon.input("input", shape: {nil, 28, 28, 1})
                           |> Axon.conv(32, kernel_size: 3, padding: :same, activation: :relu, name: "conv1")
                           |> Axon.max_pool(kernel_size: 2, name: "pool1")
                           |> Axon.conv(64, kernel_size: 3, padding: :same, activation: :relu, name: "conv2")

          {_init_fun, truncated_predict_fun} = Axon.build(truncated_model, mode: :inference)
          activations = truncated_predict_fun.(params, %{"input" => input})
          {:ok, activations}

        _ ->
          {:error, "Layer #{layer_name} not supported for visualization"}
      end
    rescue
      error -> {:error, "#{inspect(error)}"}
    catch
      error -> {:error, "#{inspect(error)}"}
    end
  end

  @doc """
  Displays feature maps from a convolutional layer activation.

  ## Parameters

  - `activations`: Tensor of activations from a conv layer, shape {1, height, width, channels}
  - `layer_name`: Name of the layer for display purposes
  - `max_filters`: Maximum number of filters/channels to display
  """
  @spec display_feature_maps(Nx.Tensor.t(), String.t(), integer()) :: :ok
  def display_feature_maps(activations, layer_name, max_filters) do
    {batch, height, width, channels} = Nx.shape(activations)

    IO.puts("\n#{IO.ANSI.yellow()}=== Feature Maps from #{layer_name} ===#{IO.ANSI.reset()}")
    IO.puts("Shape: {#{batch}, #{height}, #{width}, #{channels}} (batch, height, width, channels)")

    # Display up to max_filters feature maps
    num_filters_to_show = min(channels, max_filters)

    for filter_idx <- 0..(num_filters_to_show - 1) do
      IO.puts("\n#{IO.ANSI.green()}--- Filter #{filter_idx + 1}/#{channels} ---#{IO.ANSI.reset()}")

      # Extract single filter activation: {1, height, width, 1}
      filter_activation = activations |> Nx.slice_along_axis(filter_idx, 1, axis: 3)

      # Convert to 2D for visualization: {height, width}
      filter_2d = filter_activation |> Nx.squeeze(axes: [0, 3])

      display_feature_map_ascii(filter_2d)
    end

    :ok
  end

  # Private function to display a 2D feature map as ASCII art
  defp display_feature_map_ascii(feature_map) do
    {height, width} = Nx.shape(feature_map)

    # Normalize the feature map to 0-1 range for display
    min_val = Nx.reduce_min(feature_map)
    max_val = Nx.reduce_max(feature_map)

    normalized = if Nx.equal(min_val, max_val) |> Nx.to_number() == 1 do
      Nx.broadcast(0.5, {height, width})
    else
      Nx.subtract(feature_map, min_val)
      |> Nx.divide(Nx.subtract(max_val, min_val))
    end

    # Convert to list for processing
    feature_list = Nx.to_list(normalized)

    # Display using ASCII characters (similar to MachineLearning.Mnist.inspect)
    chars = " .:-=+*#%@"

    Enum.each(feature_list, fn row ->
      row_str = Enum.map(row, fn pixel ->
        char_index = min(trunc(pixel * (String.length(chars) - 1)), String.length(chars) - 1)
        String.at(chars, char_index)
      end) |> Enum.join("")

      IO.puts(row_str)
    end)

    # Show activation statistics
    mean_activation = Nx.mean(feature_map) |> Nx.to_number() |> Float.round(4)
    max_activation = Nx.to_number(max_val) |> Float.round(4)
    min_activation = Nx.to_number(min_val) |> Float.round(4)

    IO.puts("Stats: Min=#{min_activation}, Max=#{max_activation}, Mean=#{mean_activation}")
  end

  # Private function to display an image tensor as ASCII
  defp display_image_tensor(image_tensor) do
    # Assume input is {1, 28, 28, 1}, extract the 2D image
    image_2d = image_tensor |> Nx.squeeze(axes: [0, 3])

    # Use the same display logic as feature maps but with different character mapping
    image_list = Nx.to_list(image_2d)

    # Characters for grayscale display (darker to lighter)
    chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

    Enum.each(image_list, fn row ->
      row_str = Enum.map(row, fn pixel ->
        # Clamp pixel value between 0 and 1
        normalized_pixel = max(0, min(1, pixel))
        char_index = min(trunc(normalized_pixel * (String.length(chars) - 1)), String.length(chars) - 1)
        String.at(chars, char_index)
      end) |> Enum.join("")

      IO.puts(row_str)
    end)
  end

  @doc """
  Inspects misclassified images from the CNN test data.

  ## Parameters

  - `model`: The trained Axon CNN model
  - `params`: Trained model parameters
  - `test_data`: Test dataset (stream of {images, labels} batches)
  - `max_failures`: Maximum number of failures to show (default: 5)
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

  @doc """
  Shows a detailed analysis of how the CNN processes a specific image through its layers.

  This function combines prediction with feature map visualization to show the complete
  journey of an image through the CNN, including intermediate representations.

  ## Parameters

  - `model`: The trained Axon CNN model
  - `params`: Trained model parameters
  - `test_images_path`: Path to test images file
  - `test_labels_path`: Path to test labels file
  - `opts`: Analysis options
    - `:image_index` - Specific image index to analyze (default: random)
    - `:batch_index` - Index within the batch (default: random)
    - `:show_feature_maps` - Whether to show feature maps (default: true)
    - `:layers` - Layers to visualize (default: ["conv1", "pool1", "conv2", "pool2"])

  ## Examples

      iex> MachineLearning.AxonCnnMnist.analyze_prediction(
      ...>   model, trained_params,
      ...>   "./tmp/t10k-images.idx3-ubyte",
      ...>   "./tmp/t10k-labels.idx1-ubyte",
      ...>   show_feature_maps: true, layers: ["conv1", "conv2"]
      ...> )
  """
  @spec analyze_prediction(Axon.t(), map(), String.t(), String.t(), keyword()) :: :ok
  def analyze_prediction(model, params, test_images_path, test_labels_path, opts \\ []) do
    batch_size = 32
    image_index = Keyword.get(opts, :image_index, nil)
    batch_index = Keyword.get(opts, :batch_index, nil)
    show_feature_maps = Keyword.get(opts, :show_feature_maps, true)
    layers = Keyword.get(opts, :layers, ["conv1", "pool1", "conv2", "pool2"])

    test_data = MachineLearning.Mnist.load(test_images_path, test_labels_path, batch_size)

    # Select batch and image
    {images, labels} = if image_index do
      Enum.at(test_data, image_index)
    else
      Enum.random(test_data)
    end

    {batch_size, _} = Nx.shape(images)
    index = batch_index || Enum.random(0..(batch_size - 1))

    # Extract single image
    single_image = images |> Nx.slice_along_axis(index, 1, axis: 0)
    single_label = labels |> Nx.slice_along_axis(index, 1, axis: 0)

    # Make prediction
    predicted_probs = predict(model, params, single_image)
    predicted_class = predicted_probs |> Nx.argmax(axis: -1) |> Nx.squeeze() |> Nx.to_number()
    confidence = predicted_probs |> Nx.to_list() |> List.first() |> Enum.at(predicted_class) |> Float.round(4)

    # Get expected class
    expected_class = single_label |> Nx.argmax(axis: -1) |> Nx.squeeze() |> Nx.to_number()

    IO.puts("\n#{IO.ANSI.cyan()}=== CNN Prediction Analysis ===#{IO.ANSI.reset()}")

    # Show original image using MachineLearning.Mnist.inspect format
    {display, _} = MachineLearning.Mnist.inspect(images, labels, index)
    IO.puts(display)

    # Show prediction results
    IO.puts("#{IO.ANSI.green()}Expected: #{expected_class}#{IO.ANSI.reset()}")

    if predicted_class == expected_class do
      IO.puts("#{IO.ANSI.green()}Predicted: #{predicted_class} ✓#{IO.ANSI.reset()}")
    else
      IO.puts("#{IO.ANSI.red()}Predicted: #{predicted_class} ✗#{IO.ANSI.reset()}")
    end

    IO.puts("#{IO.ANSI.yellow()}Confidence: #{confidence * 100}%#{IO.ANSI.reset()}")

    # Show all class probabilities
    IO.puts("\n#{IO.ANSI.magenta()}Class Probabilities:#{IO.ANSI.reset()}")
    predicted_probs
    |> Nx.to_list()
    |> List.first()
    |> Enum.with_index()
    |> Enum.each(fn {prob, class} ->
      bar_length = trunc(prob * 20)
      bar = String.duplicate("█", bar_length) <> String.duplicate("░", 20 - bar_length)
      color = if class == expected_class, do: IO.ANSI.green(), else: ""
      reset = if class == expected_class, do: IO.ANSI.reset(), else: ""
      IO.puts("  #{color}#{class}: #{bar} #{Float.round(prob * 100, 1)}%#{reset}")
    end)

    # Show feature maps if requested
    if show_feature_maps do
      IO.puts("\n#{IO.ANSI.cyan()}=== Feature Map Analysis ===#{IO.ANSI.reset()}")

      # Reshape single image for CNN processing
      cnn_input = case Nx.shape(single_image) do
        {1, 784} -> Nx.reshape(single_image, {1, 28, 28, 1})
        _ -> single_image
      end

      inspect_feature_maps(model, params, cnn_input,
        layers: layers,
        max_filters: 6,
        show_original: false
      )
    end

    :ok
  end

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

  # Private function to display a single CNN misclassification
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
    IO.puts("#{String.duplicate("=", 35)}")
  end

  @doc """
  Compares the performance of CNN vs fully-connected model.

  Trains both models on the same data and compares their accuracy and training time.

  ## Parameters

  - `train_images_path`: Path to training images file
  - `train_labels_path`: Path to training labels file
  - `test_images_path`: Path to test images file
  - `test_labels_path`: Path to test labels file
  - `opts`: Training options

  ## Examples

      iex> comparison = MachineLearning.AxonCnnMnist.compare_models(
      ...>   "./tmp/train-images.idx3-ubyte",
      ...>   "./tmp/train-labels.idx1-ubyte",
      ...>   "./tmp/test-images-idx3-ubyte",
      ...>   "./tmp/test-labels-idx1-ubyte",
      ...>   epochs: 5
      ...> )
  """
  @spec compare_models(String.t(), String.t(), String.t(), String.t(), keyword()) :: map()
  def compare_models(
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
        opts \\ []
      ) do
    IO.puts("#{IO.ANSI.cyan()}=== Model Comparison: CNN vs Fully-Connected ===#{IO.ANSI.reset()}")

    # Train CNN model
    IO.puts("\n#{IO.ANSI.yellow()}Training CNN model...#{IO.ANSI.reset()}")

    cnn_result =
      train_and_evaluate(
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
        opts
      )

    # Train fully-connected model
    IO.puts("\n#{IO.ANSI.yellow()}Training Fully-Connected model...#{IO.ANSI.reset()}")

    fc_result =
      MachineLearning.AxonMnist.train_and_evaluate(
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
        opts
      )

    # Compare results
    IO.puts("\n#{IO.ANSI.cyan()}=== COMPARISON RESULTS ===#{IO.ANSI.reset()}")

    cnn_accuracy = get_accuracy(cnn_result.test_results)
    fc_accuracy = get_accuracy(fc_result.test_results)

    IO.puts("#{IO.ANSI.green()}CNN Model:#{IO.ANSI.reset()}")
    IO.puts("  - Accuracy: #{Float.round(cnn_accuracy * 100, 2)}%")
    IO.puts("  - Training time: #{cnn_result.training_time}s")

    IO.puts("#{IO.ANSI.blue()}Fully-Connected Model:#{IO.ANSI.reset()}")
    IO.puts("  - Accuracy: #{Float.round(fc_accuracy * 100, 2)}%")
    IO.puts("  - Training time: #{fc_result.training_time}s")

    accuracy_improvement = (cnn_accuracy - fc_accuracy) * 100

    IO.puts(
      "\n#{IO.ANSI.magenta()}CNN Accuracy Improvement: #{Float.round(accuracy_improvement, 2)} percentage points#{IO.ANSI.reset()}"
    )

    %{
      cnn: cnn_result,
      fully_connected: fc_result,
      cnn_accuracy: cnn_accuracy,
      fc_accuracy: fc_accuracy,
      accuracy_improvement: accuracy_improvement
    }
  end

  # Helper function to extract accuracy from test results
  defp get_accuracy(test_results) do
    case test_results do
      %{0 => %{"accuracy" => accuracy}} -> accuracy
      %{accuracy: accuracy} -> accuracy
      _ -> 0.0
    end
  end
end
