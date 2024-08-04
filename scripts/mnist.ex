model = MachineLearning.Model.init([784, 128, 10], 0.01)
set = MachineLearning.Mnist.load("./tmp/train-images-idx3-ubyte", "./tmp/train-labels-idx1-ubyte", 30)

IO.puts("Training...")
model = set
  |> Enum.with_index()
  |> Enum.reduce(model, fn {{image, expected}, index}, model ->
    if rem(index, 100) == 0 do
      IO.puts("Index #{index}")
      IO.puts("Calculating loss...")
      MachineLearning.Model.loss(model, image, expected)
      |> Nx.to_number()
      |> IO.inspect()

      IO.puts("Accuracy ...")
      MachineLearning.Model.accuracy(model, image, expected)
      |> Nx.to_number()
      |> IO.inspect()
    end

  MachineLearning.Model.train(model, image, expected)
end)

IO.puts("Predicting...")
{single_image, expected} = MachineLearning.Mnist.load("./tmp/train-images-idx3-ubyte", "./tmp/train-labels-idx1-ubyte", 30) |> Enum.at(0)
MachineLearning.Model.predict(model, single_image)
|> IO.inspect()

expected |> IO.inspect()

model
