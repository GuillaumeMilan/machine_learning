defmodule MachineLearning.Model do
  defstruct layers: nil

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
end
