defmodule MachineLearning.Mnist do
  @moduledoc """
  Maniuplate MNIST dataset

  Inspired from https://github.com/elixir-nx/nx/blob/main/exla/examples/mnist.exs
  """

  @doc """
  Download the MNIST dataset into the ./tmp folder.

    iex> MachineLearning.Mnist.download!()
  """
  @spec download!() :: :ok
  def download!() do
    File.rm_rf!("./tmp")
    File.mkdir_p!("./tmp")

    {_, 0} =
      System.cmd("wget", [
        "https://storage.googleapis.com/kaggle-data-sets/102285/242592/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251214%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251214T220517Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=722881189a6d825490e46c7da6179f88a2351adf4ea003dbbde496216097dc1f783bdce898f813e808503a6d25a1baf4f9b6636e7e65feeb34c8d87f32f470dc273f6930c28a8cb18efdcd9e35a51507639858f3b38982bf491cfe0f639527ff055b21f5359c64cddd0e204893e030b8d7c1683ef1a979d999ecc7ad6f2124e77aae2c1ec82477ed078208cc6a7d2b939ef0c724b7eeb9872a7096135c71acc65495ec3430e04582997882e78cd82e7826069f7fbf07d7269b68806b0a88149bb2019ad59738f6368afa4cb5daf54139048dec5532bf0851f00b6d32fdc3a9dbc2e692a64e67f3a496b40429be3302b6ba6f8da52be20d3a6895181fd07ec7ff",
        "-O",
        "./tmp/archive.zip"
      ])

    {_, 0} = System.cmd("unzip", ["./tmp/archive.zip", "-d", "./tmp/"])
    :ok
  end

  @doc """
  Load the MNIST dataset from the given images and labels paths.

    iex> MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte")
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

    iex> {images, labels} = MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte") |> Enum.at(1)
    iex> MachineLearning.Mnist.inspect(images, labels, 0)
  """
  @spec inspect(images_batch :: Nx.Tensor.t(), labels_batch :: Nx.Tensor.t(), index :: integer) ::
          {String.t(), Integer.t()}
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

    iex> set = MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte", 30)
    iex> MachineLearning.Mnist.check(model, set)
  """
  @spec check(MachineLearning.Network.t(), Enumerable.t()) :: :ok
  def check(model, set) do
    set
    |> Enum.random()
    |> then(fn {images, labels} ->
      {batch_size, _} = Nx.shape(images)
      index = Enum.random(0..(batch_size - 1))

      predicted =
        MachineLearning.Network.predict(model, images)
        |> Nx.argmax(axis: -1)
        |> Nx.to_list()
        |> Enum.at(index)

      {display, expected} = MachineLearning.Mnist.inspect(images, labels, index)
      IO.puts(display)
      IO.puts("Expected: #{expected}")
      IO.puts("Predicted: #{predicted}")
    end)
  end

  def inspect_first_layer(model, index) do
    weights = model.layers |> List.first() |> Map.get(:weights)
    max_value = Nx.reduce_max(weights) |> Nx.to_number()
    min_value = Nx.reduce_min(weights) |> Nx.to_number()

    weights
    |> Nx.add(-min_value)
    |> Nx.divide(max_value - min_value)
    |> Nx.multiply(92 * 2)
    |> Nx.subtract(92)
    |> Nx.transpose()
    |> Nx.to_list()
    |> Enum.at(index)
    |> Enum.chunk_every(28)
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

  defp gray_scale_ascii(scale) do
    #  `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    |> String.graphemes()
    |> Enum.at(scale)
  end

  defp color(:plus), do: 23
  defp color(:minus), do: 1

  defp label_to_human(label) do
    label
    |> Enum.with_index()
    |> Enum.find(fn {v, _} -> v == 1 end)
    |> elem(1)
  end
end
