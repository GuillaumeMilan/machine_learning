defmodule MachineLearning.Layer do
  @derive {Nx.Container,containers: [:weights, :biases]}

  @enforce_keys [:weights, :biases]
  defstruct weights: nil, biases: nil
  @type t :: %__MODULE__{weights: Nx.Tensor.t(), biases: Nx.Tensor.t()}

  @doc """
  Create a new layer with the given weights and biases.
  """
  @spec new(Nx.Tensor.t(), Nx.Tensor.t()) :: t()
  def new(weights, biases) do
    %__MODULE__{weights: weights, biases: biases}
  end

  @doc """
  Update the layer with the given gradient and step.
  iex> MachineLearning.Layer.update(layer, gradient, 0.01)
  """
  @spec update(t(), t(), float) :: t()
  def update(layer, gradient, step) do
    %__MODULE__{
      weights: Nx.subtract(layer.weights, Nx.multiply(gradient.weights, step)),
      biases: Nx.subtract(layer.biases, Nx.multiply(gradient.biases, step))
    }
  end
end
