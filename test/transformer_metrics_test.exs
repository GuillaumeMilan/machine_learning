defmodule MachineLearning.TransformerMetricsTest do
  use ExUnit.Case
  doctest MachineLearning.Transformer

  alias MachineLearning.Transformer
  alias MachineLearning.BytePairEncoding.Token

  describe "epoch tracking and metrics" do
    @tag :tmp_dir
    test "saves current epoch number to current_epoch.txt", %{tmp_dir: tmp_dir} do
      model_dir = Path.join(tmp_dir, "test_model_1")

      # Create a model
      tokens = Enum.map(0..50, fn i -> Token.new("tok_#{i}") end)

      _model =
        Transformer.create(
          tokens: tokens,
          save_dir: model_dir,
          embed_dim: 64,
          max_seq_len: 32,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 128
        )

      # Add training data with longer sequences
      Transformer.add_training_data(
        model_dir,
        "dataset_1",
        token_sequences: [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        batch_size: 2,
        seq_len: 8
      )

      # Train for 1 epoch
      _trained =
        Transformer.train(
          model_dir,
          "dataset_1",
          epochs: 1,
          learning_rate: 0.001
        )

      # Verify current_epoch.txt was created
      epoch_file = Path.join(model_dir, "current_epoch.txt")
      assert File.exists?(epoch_file), "current_epoch.txt should exist"

      # Verify it contains epoch 1
      epoch_content = File.read!(epoch_file) |> String.trim()
      assert epoch_content == "1", "Should have saved epoch 1"
    end

    @tag :tmp_dir
    test "saves metrics to CSV file", %{tmp_dir: tmp_dir} do
      model_dir = Path.join(tmp_dir, "test_model_2")

      # Create a model
      tokens = Enum.map(0..50, fn i -> Token.new("tok_#{i}") end)

      _model =
        Transformer.create(
          tokens: tokens,
          save_dir: model_dir,
          embed_dim: 64,
          max_seq_len: 32,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 128
        )

      # Add training data with longer sequences
      Transformer.add_training_data(
        model_dir,
        "dataset_1",
        token_sequences: [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        batch_size: 2,
        seq_len: 8
      )

      # Train for 1 epoch
      _trained =
        Transformer.train(
          model_dir,
          "dataset_1",
          epochs: 1,
          learning_rate: 0.001
        )

      # Verify metrics.csv was created
      metrics_file = Path.join(model_dir, "metrics.csv")
      assert File.exists?(metrics_file), "metrics.csv should exist"

      # Verify it has header and at least one data row
      content = File.read!(metrics_file)
      lines = String.split(content, "\n") |> Enum.filter(&(&1 != ""))

      assert length(lines) >= 2, "Should have header + at least 1 data row"
      assert String.starts_with?(hd(lines), "epoch,loss,accuracy"), "Should have CSV header"

      # Verify second line has epoch 1
      second_line = Enum.at(lines, 1)
      assert String.starts_with?(second_line, "1,"), "First data row should be epoch 1"
    end

    @tag :tmp_dir
    test "resumes training from saved epoch", %{tmp_dir: tmp_dir} do
      model_dir = Path.join(tmp_dir, "test_model_3")

      # Create a model
      tokens = Enum.map(0..50, fn i -> Token.new("tok_#{i}") end)

      _model =
        Transformer.create(
          tokens: tokens,
          save_dir: model_dir,
          embed_dim: 64,
          max_seq_len: 32,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 128
        )

      # Add training data with longer sequences
      Transformer.add_training_data(
        model_dir,
        "dataset_1",
        token_sequences: [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        batch_size: 2,
        seq_len: 8
      )

      # Train for 2 epochs in first session
      _trained1 =
        Transformer.train(
          model_dir,
          "dataset_1",
          epochs: 2,
          learning_rate: 0.001
        )

      # Verify epoch 2 was saved
      epoch_file = Path.join(model_dir, "current_epoch.txt")
      epoch_content1 = File.read!(epoch_file) |> String.trim()
      assert epoch_content1 == "2", "Should have saved epoch 2"

      # Train for 2 more epochs in second session
      _trained2 =
        Transformer.train(
          model_dir,
          "dataset_1",
          epochs: 2,
          learning_rate: 0.001
        )

      # Verify epoch 4 was saved
      epoch_content2 = File.read!(epoch_file) |> String.trim()
      assert epoch_content2 == "4", "Should have saved epoch 4"

      # Verify metrics.csv has 4 epochs
      metrics_file = Path.join(model_dir, "metrics.csv")
      content = File.read!(metrics_file)
      lines = String.split(content, "\n") |> Enum.filter(&(&1 != ""))

      assert length(lines) == 5, "Should have header + 4 epochs"

      # Verify epochs are 1, 2, 3, 4
      epochs =
        lines
        |> tl()
        |> Enum.map(fn line ->
          String.split(line, ",") |> hd() |> String.to_integer()
        end)

      assert epochs == [1, 2, 3, 4], "Epochs should be sequential"
    end

    @tag :tmp_dir
    test "metrics file has correct format", %{tmp_dir: tmp_dir} do
      model_dir = Path.join(tmp_dir, "test_model_4")

      # Create a model
      tokens = Enum.map(0..50, fn i -> Token.new("tok_#{i}") end)

      _model =
        Transformer.create(
          tokens: tokens,
          save_dir: model_dir,
          embed_dim: 64,
          max_seq_len: 32,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 128
        )

      # Add training data with longer sequences
      Transformer.add_training_data(
        model_dir,
        "dataset_1",
        token_sequences: [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
          [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ],
        batch_size: 2,
        seq_len: 8
      )

      # Train for 1 epoch
      _trained =
        Transformer.train(
          model_dir,
          "dataset_1",
          epochs: 1,
          learning_rate: 0.001
        )

      # Read metrics
      metrics_file = Path.join(model_dir, "metrics.csv")
      content = File.read!(metrics_file)
      lines = String.split(content, "\n") |> Enum.filter(&(&1 != ""))

      # Parse header
      header = hd(lines)
      assert header == "epoch,loss,accuracy", "Header should match expected format"

      # Parse data row
      data_row = Enum.at(lines, 1)
      fields = String.split(data_row, ",")

      assert length(fields) == 3, "Each row should have 3 fields"

      # Verify epoch is a number
      epoch_str = hd(fields)
      {epoch_num, ""} = Integer.parse(epoch_str)
      assert epoch_num == 1, "First column should be epoch number"

      # Loss and accuracy can be "N/A" or numbers
      loss = Enum.at(fields, 1)
      accuracy = Enum.at(fields, 2)

      assert loss in ["N/A"] or is_float_or_int_string(loss),
             "Loss should be N/A or a number"

      assert accuracy in ["N/A"] or is_float_or_int_string(accuracy),
             "Accuracy should be N/A or a number"
    end
  end

  # Helper function to check if a string represents a float or integer
  defp is_float_or_int_string(str) do
    case Float.parse(str) do
      {_num, ""} -> true
      _other -> false
    end
  end
end
