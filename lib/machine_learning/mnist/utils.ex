defmodule MachineLearning.Mnist.Utils do
  def get_first_entry(stream) do
    stream
    |> Enum.at(0)
    |> then(fn {image, label} ->
      {
        Nx.reshape(image, {784}),
        Nx.reshape(label, {10})
      }
    end)
  end
end
