defmodule MachineLearning.Transformer.Scripts do
  def train_few_steps_models(original_model, corpus_path, iteration_count) do
    # Placeholder for training few models script

    1..iteration_count
    |> Enum.reduce(original_model, fn iteration, acc_model ->
      model =
        MachineLearning.TransformerTraining.run_on_model(acc_model, %{
          epoch: 3,
          corpus_dir: corpus_path,
          sample_size: 50,
          log_level: :debug
        })

      IO.puts("Completed training iteration #{iteration}")

      evaluate_model(model)
      model
    end)
  end

  def evaluate_model(model) do
    # Placeholder for model evaluation logic
    IO.puts("Evaluating model...")

    original_input =
      """
      Filename: /Users/guillaumemilan/Documents/Projects/perso/machine_learning/lib/machine_learning/corpus.ex
      ---------------------------------------------
      defmodule MachineLearning.Corpus do
        @moduledoc \"""
        Provides functions
      """
      |> String.trim()
      |> then(&(&1 <> " "))

    1..5
    |> Enum.reduce(original_input, fn _, input ->
      MachineLearning.TransformerTraining.predict(model, input)
    end)
    |> IO.puts()
  end
end
