defmodule MachineLearning.Model do
  alias MachineLearning.Layer

  defstruct layers: nil

  @type t :: %__MODULE__{layers: list(MachineLearning.Layer.t())}
  @type execution_with_activation :: {Nx.tensor(), list({Nx.tensor(), Nx.tensor()})}

  @doc """
  Initialize the model with the given layer sizes.

  iex> MachineLearning.Model.init([784, 16, 16, 10])
  """
  def init(layer_sizes) do
    Enum.zip(layer_sizes, Enum.drop(layer_sizes, 1))
    |> Enum.map(fn {input_size, output_size} ->
      MachineLearning.Layer.init(input_size, output_size)
    end)
    |> then(&%__MODULE__{layers: &1})
  end

  @doc """
  Execute the model on the given input.
  iex> MachineLearning.Model.execute(model, input)
  """
  def execute(%__MODULE__{layers: layers}, input) do
    Enum.reduce(layers, input, fn layer, activation ->
      MachineLearning.Layer.execute(layer, activation)
    end)
  end

  @doc """
  Train the model on the given batch of inputs and expected outputs.
  """
  @spec train(t(), list({input :: Nx.tensor(), expected :: Nx.tensor()})) :: t()
  def train(model, batch) do
    batch
    # Calculate the cost gradient for each input
    |> Enum.map(fn {input, expected} ->
      {activations, outputs} = execute_with_activations(model, input)
      gradient = MachineLearning.Cost.gradient(model, {activations, outputs}, expected)
      value = MachineLearning.Cost.value(activations, expected)
      {gradient, value}
    end)
    |> then(&mean_gradients(&1))
    |> then(fn {gradient, value} ->
      %__MODULE__{
        layers: Enum.zip(model.layers, gradient)
        |> Enum.map(fn {layer, gradient} ->
          %Layer{
            biases: update_from_gradient(layer.biases, gradient.biases, value),
            weights: update_from_gradient(layer.weights, gradient.weights, value)
          }
        end)
      }
    end)
  end

  @doc """
  Execute the model on the given input and keep all the activations.

  This is useful for back propagation.
  iex> MachineLearning.Model.execute_with_activations(model, input)
  """
  @spec execute_with_activations(t(), Nx.tensor()) :: execution_with_activation()
  def execute_with_activations(%__MODULE__{layers: layers}, input) do
    Enum.reduce(layers, {input, []}, fn layer, {last_activation, activations} ->
      next_activation = MachineLearning.Layer.execute(layer, last_activation)
      {next_activation, [{next_activation, last_activation} | activations]}
    end)
  end

  defp mean_gradients(gradients) do
    [first_gradient| rest] = gradients
    rest
    |> Enum.reduce(first_gradient, &sum_gradients(&1, &2))
    |> then(fn {layers, value} ->
      {
        Enum.map(layers, fn layer ->
          %Layer{biases: Nx.divide(layer.biases, length(gradients)), weights: Nx.divide(layer.weights, length(gradients))}
        end),
        Nx.divide(value, length(gradients))
      }
    end)
  end

  defp sum_gradients({layers1, value1}, {layers2, value2}) do
    {
      Enum.reduce(Enum.zip(layers1, layers2), [], fn {layer1, layer2}, acc ->
        [sum_layers(layer1, layer2) | acc]
      end)
      |> Enum.reverse(),
      Nx.add(value1, value2)
    }
  end

  defp sum_layers(%Layer{biases: biases1, weights: weights1}, %Layer{biases: biases2, weights: weights2}) do
    %Layer{biases: Nx.add(biases1, biases2), weights: Nx.add(weights1, weights2)}
  end

  defp update_from_gradient(layer, gradient, value) do
    layer
    |> Nx.subtract(Nx.divide(value, gradient))
  end
end
