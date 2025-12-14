defmodule MachineLearning.Layer do
  @derive {Nx.Container, containers: [:weights, :biases]}

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

  @doc """
  Inspect layer weights on the format of an image.

    iex> MachineLearning.Layer.inspect_weigths(layer) |> IO.puts()
  """
  @spec inspect_weigths(t()) :: String.t()
  def inspect_weigths(%__MODULE__{weights: weights}) do
    max_value = Nx.reduce_max(weights) |> Nx.to_number()
    min_value = Nx.reduce_min(weights) |> Nx.to_number()

    weights
    |> Nx.add(-min_value)
    |> Nx.divide(max_value - min_value)
    |> Nx.multiply(92 * 2)
    |> Nx.subtract(92)
    |> Nx.to_list()
    |> Enum.map(fn row ->
      row
      |> Enum.map(fn col ->
        sign = if col >= 0, do: :plus, else: :minus
        symbole = gray_scale_ascii(abs(round(col)))
        "#{IO.ANSI.color(color(sign))}#{symbole}"
      end)
      |> Enum.join()
    end)
    |> Enum.join("\n")
    |> then(&"#{IO.ANSI.black_background()}#{&1}#{IO.ANSI.reset()}")
  end

  defp color(:plus), do: 23
  defp color(:minus), do: 1

  defp gray_scale_ascii(scale) do
    #  `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    |> String.graphemes()
    |> Enum.at(scale)
  end
end
