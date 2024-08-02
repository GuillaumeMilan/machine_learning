defmodule MachineLearning.Layer do
  defstruct weights: nil, biases: nil

  def init(input_size, output_size) do
    %__MODULE__{
      weights: init_weights(input_size, output_size),
      biases: init_bias(output_size)
    }
  end

  @doc"""
  Execute the layer to get the next activation layer.

  iex> MachineLearning.Layer.execute(layer, activation)
  """
  def execute(%__MODULE__{weights: weights, biases: biases}, activation) do
    Nx.dot(weights, activation)
    |> Nx.add(biases)
    |> Nx.sigmoid()
  end

  defp init_weights(input_size, output_size) do
    Nx.tensor(
      1..output_size
      |> Enum.map(fn _ ->
        1..input_size
        |> Enum.map(fn _ -> :rand.uniform() end)
      end)
    )
  end

  defp init_bias(output_size) do
    Nx.tensor(1..output_size |> Enum.map(fn _ -> :rand.uniform() end))
  end


end
