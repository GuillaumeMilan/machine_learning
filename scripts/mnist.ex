model = MachineLearning.Network.init([784, 128, 10], 0.01)

set =
  MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte", 10)

IO.puts("Training...")

model =
  set
  |> Enum.with_index()
  |> Enum.reduce(model, fn {{image, expected}, index}, model ->
    if rem(index, 100) == 0 do
      IO.puts("Index #{index}")

      MachineLearning.Network.loss(model, image, expected)
      |> Nx.to_number()
      |> IO.inspect(label: "Loss")

      MachineLearning.Network.accuracy(model, image, expected)
      |> Nx.to_number()
      |> IO.inspect(label: "Accuracy")

      # MachineLearning.Network.gradient(model, image, expected)
      # |> IO.inspect(label: "Gradient")
    end

    MachineLearning.Network.train(model, image, expected)
  end)

IO.puts("Predicting...")

{single_image, expected} =
  MachineLearning.Mnist.load("./tmp/train-images.idx3-ubyte", "./tmp/train-labels.idx1-ubyte", 30)
  |> Enum.at(0)

MachineLearning.Network.predict(model, single_image)
|> IO.inspect()

expected |> IO.inspect()

model
