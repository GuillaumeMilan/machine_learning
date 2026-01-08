model = MachineLearning.Transformer.load("models/small")

files =
  File.ls!("./lib/machine_learning/")
  |> Enum.filter(&String.ends_with?(&1, ".ex"))
  |> Enum.map(&"./lib/machine_learning/#{&1}")
  |> Enum.shuffle()
  |> Enum.take(5)

prediction_size = 10
context_window = 128

pieces =
  files
  |> Enum.map(fn filename ->
    tokens =
      File.read!(filename)
      |> then(&MachineLearning.Tokenizer.encode(model.tokenizer, &1))

    # Randomize the window of tokens inside the file from which we take a slice

    start_index = Enum.random(0..max(0, length(tokens) - context_window - prediction_size))
    end_index = start_index + context_window + prediction_size

    Enum.slice(tokens, start_index..(end_index - 1))
  end)

inputs =
  pieces
  |> Enum.map(fn tokens ->
    input_tokens = Enum.slice(tokens, 0, length(tokens) - prediction_size)
    MachineLearning.Tokenizer.decode(model.tokenizer, input_tokens)
  end)

expected_predictions =
  pieces
  |> Enum.map(fn tokens ->
    expected_tokens = Enum.slice(tokens, length(tokens) - prediction_size, prediction_size)
    MachineLearning.Tokenizer.decode(model.tokenizer, expected_tokens)
  end)

outputs =
  MachineLearning.Transformer.batch_predict(
    model,
    Enum.map(inputs, fn input -> input end),
    max_length: prediction_size
  )

outputs
|> Enum.zip(files)
|> Enum.zip(inputs)
|> Enum.zip(expected_predictions)
|> Enum.each(fn {{{output, filename}, inputs}, expected_prediction} ->
  predicted = output |> String.slice(String.length(inputs), String.length(output))
  output = inputs <> IO.ANSI.bright() <> predicted <> IO.ANSI.reset()
  expected_output = inputs <> IO.ANSI.bright() <> expected_prediction <> IO.ANSI.reset()
  IO.puts("=== Prediction ===")
  IO.puts("File: #{filename}")
  IO.puts("------------------")
  IO.puts(output)
  IO.puts("------------------")
  IO.puts("=== Expected ===")
  IO.puts(expected_output)
  IO.puts("==================\n")
end)

# results = MachineLearning.Transformer.test(
#   "models/small",
#   "test_elixir",
#   params_version: "latest"
# )
