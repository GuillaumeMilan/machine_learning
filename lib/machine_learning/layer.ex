defmodule MachineLearning.Layer do
  @derive {Nx.Container,containers: [:weights, :biases]}

  @enforce_keys [:weights, :biases]
  defstruct weights: nil, biases: nil

  def init(input_size, output_size) do
    %__MODULE__{
      weights: init_weights(input_size, output_size),
      biases: init_bias(output_size)
    }
  end

  def update(layer, gradient, step) do
    %__MODULE__{
      weights: Nx.subtract(layer.weights, Nx.multiply(gradient.weights, step)),
      biases: Nx.subtract(layer.biases, Nx.multiply(gradient.biases, step))
    }
  end

  @doc"""
  Execute the layer to get the next activation layer.

  iex> MachineLearning.Layer.execute(layer, activation)
  """
  def execute(%__MODULE__{weights: weights, biases: biases}, activation) do
    try do
      activation
      |> Nx.dot(weights)
      |> Nx.add(biases)
      |> Nx.sigmoid()
    catch
      _,_ -> raise "Error in Layer execution #{inspect weights} #{inspect biases} #{inspect activation}"
    end
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
