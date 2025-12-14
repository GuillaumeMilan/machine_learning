defmodule MachineLearning.Network do
  @moduledoc """
  Define a fully connected neural network.
  """
  alias MachineLearning.Layer
  import Nx.Defn

  @derive {Nx.Container, containers: [:layers]}
  defstruct layers: nil, execute_fun: nil, loss_fun: nil, grad_fun: nil, step: nil

  @type t :: %__MODULE__{
          layers: list(MachineLearning.Layer.t()),
          step: float(),
          execute_fun: function(),
          loss_fun: function(),
          grad_fun: function()
        }
  @type execution_with_activation :: {Nx.Tensor.t(), list({Nx.Tensor.t(), Nx.Tensor.t()})}

  @doc """
  Initialize the network with the given layer sizes.

    iex> MachineLearning.Network.init([784, 16, 16, 10])
  """
  def init(layer_sizes, step \\ 0.01) do
    key = Nx.Random.key(42)
    nb_layers = Enum.count(layer_sizes) - 1

    Enum.zip(layer_sizes, Enum.drop(layer_sizes, 1))
    |> Enum.with_index(1)
    |> Enum.reduce({key, []}, fn {{input_size, output_size}, index}, {key, acc} ->
      {weights, key} = Nx.Random.normal(key, 0.0, 0.1, shape: {input_size, output_size})
      {biases, key} = Nx.Random.normal(key, 0.0, 0.1, shape: {output_size})
      {key, [{:"l#{index}", Layer.new(weights, biases)} | acc]}
    end)
    |> elem(1)
    |> Map.new()
    |> then(&%__MODULE__{layers: &1, step: step})
    |> then(&Map.put(&1, :execute_fun, EXLA.jit(execute_fun(nb_layers))))
    |> then(&Map.put(&1, :loss_fun, EXLA.jit(loss_fun(execute_fun(nb_layers)))))
    |> then(&Map.put(&1, :grad_fun, EXLA.jit(grad_fun(loss_fun(execute_fun(nb_layers))))))
  end

  @doc """
  Run the network to predict the output of the given input.

    iex> MachineLearning.Network.predict(network, input)
  """
  @spec predict(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def predict(network, input) do
    network
    |> network.execute_fun.(input)
  end

  @doc """
  Calculate the loss of the network with the given activations and expected output.
  Activations and expected output must be respectively of shape
  `{batch_size, input_size}` and `{batch_size, output_size}`.

    iex> MachineLearning.Network.loss(network, input, expected)
  """
  @spec loss(t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def loss(network, input, expected) do
    network
    |> network.loss_fun.(input, expected)
  end

  @doc """
  Calculate the accuracy of the network with the given input and expected output.

    iex> MachineLearning.Network.accuracy(network, input, expected)
  """
  @spec accuracy(t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def accuracy(network, activation, expected) do
    network
    |> predict(activation)
    |> Nx.argmax(axis: -1)
    |> Nx.equal(Nx.argmax(expected, axis: -1))
    |> Nx.mean()
  end

  @doc """
  Calculate the gradient of the network with the given activations and expected output.
  Activations and expected output must be respectively of shape
  `{batch_size, input_size}` and `{batch_size, output_size}`.

  Returns the gradient of the network's layers on the form of a list of gradients.

    iex> MachineLearning.Network.gradient(network, activations, expected)
  """
  @spec gradient(t(), Nx.Tensor.t(), Nx.Tensor.t()) :: list(Layer.t())
  def gradient(network, activations, expected) do
    network
    |> network.grad_fun.(activations, expected)
  end

  @doc """
  Train the network with the given input and expected output.
  activations and expected output must be respectively of shape
  `{batch_size, input_size}` and `{batch_size, output_size}`.
  Returns the updated network.

    iex> MachineLearning.Network.train(network, activations, expected)
  """
  @spec train(t(), activations :: Nx.Tensor.t(), expected :: Nx.Tensor.t()) :: t()
  def train(network, activations, expected) do
    gradient(network, activations, expected)
    |> then(&apply_gradient(network, &1))
  end

  @doc """
  Apply the given gradients to the network and return the updated network.

    iex> MachineLearning.Network.apply_gradient(network, gradient)
  """
  @spec apply_gradient(t(), t()) :: t()
  def apply_gradient(network, gradient) do
    %{
      network
      | layers:
          network.layers
          |> Map.new(fn {key, layer} ->
            {key, Layer.update(layer, gradient.layers[key], network.step)}
          end)
    }
  end

  # Function compiled when generating the network
  # They are also optimized using EXLA.jit

  defp grad_fun(loss_fun) do
    fn network, input, expected ->
      network
      |> grad(&loss_fun.(&1, input, expected))
    end
  end

  defp loss_fun(execute_fun) do
    fn network, input, expected ->
      execute_fun.(network, input)
      |> Nx.subtract(expected)
      |> Nx.pow(2)
      |> Nx.mean(axes: [-1])
      |> Nx.sum()
    end
  end

  defp execute_fun(nb_layers) do
    network_var = Macro.var(:network, nil)
    input_var = Macro.var(:input, nil)

    quote_execution(input_var, network_var, nb_layers)
    |> then(fn body ->
      quote do
        fn unquote(network_var), unquote(input_var) -> unquote(body) end
      end
    end)
    |> Code.eval_quoted()
    |> elem(0)
  end

  def quote_execution(input_var, network_var, nb_layers) do
    1..nb_layers
    |> Enum.reduce(
      input_var,
      fn index, acc ->
        quote do
          unquote(acc)
          |> Nx.dot(unquote(network_var).layers.unquote(Macro.var(:"l#{index}", nil)).weights)
          |> Nx.add(unquote(network_var).layers.unquote(Macro.var(:"l#{index}", nil)).biases)
          |> Nx.sigmoid()
        end
      end
    )
  end
end
