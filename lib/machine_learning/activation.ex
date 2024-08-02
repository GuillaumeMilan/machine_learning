defmodule MachineLearning.Activation do
  @moduledoc """
  Documentation for `MachineLearning.Activation`.
  """

  def generate_random(size) do
    Nx.tensor(1..size |> Enum.map(fn _ -> :rand.uniform() end))
  end
end
