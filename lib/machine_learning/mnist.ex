defmodule MachineLearning.Mnist do
  @moduledoc """
  Maniuplate MNIST dataset

  Inspired from https://github.com/elixir-nx/nx/blob/main/exla/examples/mnist.exs
  """

  @doc """
  Load the MNIST dataset from the given images and labels paths.

    iex> MachineLearning.Mnist.load("./tmp/train-images-idx3-ubyte", "./tmp/train-labels-idx1-ubyte")
  """
  @spec load(Path.t(), Path.t(), integer) :: Enumerable.t()
  def load(images_path, labels_path, batch_size \\ 30) do
    content = File.read!(images_path)
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> = content
    train_images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols}, names: [:batch, :input])
      |> Nx.divide(255)
      |> Nx.to_batched(batch_size)

    IO.puts("#{n_images} #{n_rows}x#{n_cols} images\n")

    content = File.read!(labels_path)
    <<_::32, n_labels::32, labels::binary>> = content

    train_labels =
      labels
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      |> Nx.to_batched(batch_size)

    IO.puts("#{n_labels} labels\n")
    Stream.zip(train_images, train_labels)
  end

  @doc """
  Inspect one entry from the MNIST dataset.

    iex> {images, labels} = MachineLearning.Mnist.load("./tmp/train-images-idx3-ubyte", "./tmp/train-labels-idx1-ubyte") |> Enum.at(1)
    iex> MachineLearning.Mnist.inspect(images, labels, 0)
  """
  @spec inspect(images_batch :: Nx.Tensor.t(), labels_batch :: Nx.Tensor.t(), index :: integer) :: {String.t(), Integer.t()}
  def inspect(images, labels, index) do
    images
    |> Nx.to_list()
    |> Enum.at(index)
    |> Enum.chunk_every(28)
    |> Enum.map(fn row ->
      row
      # Gray scale is on 0-92 while we receive 0-1
      |> Enum.map(&(&1 * 92))
      |> Enum.map(&Kernel.round/1)
      |> Enum.map(&gray_scale_ascii/1)
      |> Enum.join()
    end)
    |> Enum.join("\n")
    |> then(&{&1, labels |> Nx.to_list() |> Enum.at(index) |> label_to_human()})
  end

  @doc """
  Check a model against any random input in the data-set

    iex> set = MachineLearning.Mnist.load("./tmp/train-images-idx3-ubyte", "./tmp/train-labels-idx1-ubyte", 30)
    iex> MachineLearning.Mnist.check(model, set)
  """
  @spec check(MachineLearning.Model.t(), Enumerable.t()) :: :ok
  def check(model, set) do
    set
    |> Enum.random()
    |> then(fn {images, labels} ->
      {batch_size, _} = Nx.shape(images)
      index = Enum.random(0..batch_size - 1)
      predicted = MachineLearning.Model.predict(model, images)
      |> Nx.argmax(axis: -1)
      |> Nx.to_list()
      |> Enum.at(index)
      {display, expected} = MachineLearning.Mnist.inspect(images, labels, index)
      IO.puts(display)
      IO.puts("Expected: #{expected}")
      IO.puts("Predicted: #{predicted}")
    end)
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
