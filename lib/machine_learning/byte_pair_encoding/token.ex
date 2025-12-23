defmodule MachineLearning.BytePairEncoding.Token do
  defstruct [:value]

  @type t :: %__MODULE__{
          value: String.t()
        }

  @spec new(t() | String.t(), t() | String.t()) :: %MachineLearning.BytePairEncoding.Token{}
  def new(left, right) do
    left = to_string(left)
    right = to_string(right)
    %__MODULE__{value: "#{left}#{right}"}
  end

  def new(value) when is_binary(value) do
    %__MODULE__{value: value}
  end
end

defimpl String.Chars, for: MachineLearning.BytePairEncoding.Token do
  alias MachineLearning.BytePairEncoding.Token

  def to_string(%Token{value: value}) do
    value
  end
end
