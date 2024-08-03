defmodule MachineLearning.Mnist do
  @moduledoc """
  Maniuplate MNIST dataset

  Inspired from https://github.com/elixir-nx/nx/blob/main/exla/examples/mnist.exs
  """

  def load_mnist(images_path, labels_path) do
    content = File.read!(images_path)
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> = content
    train_images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols}, names: [:batch, :input])
      |> Nx.divide(255)
      |> Nx.to_batched(30)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    content = File.read!(labels_path)
    <<_::32, n_labels::32, labels::binary>> = content

    train_labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.to_batched(30)

    IO.puts("#{n_labels} labels\n")

    {train_images, train_labels}
  end

  @doc """
  Print batch from the MNIST dataset
  """
  @spec inspect(images_batch :: Nx.tensor, labels_batch :: Nx.tensor, index :: integer) :: {String.t(), Integer.t()}
  def inspect(images, labels, index) do
    images
    |> Nx.to_list()
    |> Enum.at(index)
    |> Enum.chunk_every(28)
    |> Enum.map(fn row ->
      row
      |> Enum.map(&Kernel.round/1)
      |> Enum.map(&gray_scale_ascii/1)
      |> Enum.join()
    end)
    |> Enum.join("\n")
    |> then(&{&1, labels |> Nx.to_list() |> Enum.at(index) |> label_to_human()})
  end

  defp gray_scale_ascii(scale) do
    #  `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    |> String.graphemes()
    |> Enum.at(scale)
  end

  defp label_to_human(label) do
    label
    |> Enum.with_index()
    |> Enum.find(fn {v, _} -> v == 1 end)
    |> elem(1)
  end
end
