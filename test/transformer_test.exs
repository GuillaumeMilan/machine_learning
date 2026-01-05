defmodule MachineLearning.TransformerTest do
  use ExUnit.Case
  doctest MachineLearning.Transformer

  alias MachineLearning.Transformer

  @test_model_dir "test/tmp/test_transformer_model"

  setup do
    # Clean up any existing test model
    if File.dir?(@test_model_dir), do: File.rm_rf!(@test_model_dir)

    # Set EXLA backend
    Nx.global_default_backend(EXLA.Backend)

    # Create dummy vocabulary tokens
    dummy_tokens = create_dummy_tokens()

    on_exit(fn ->
      # Clean up after tests
      if File.dir?(@test_model_dir), do: File.rm_rf!(@test_model_dir)
    end)

    {:ok, tokens: dummy_tokens}
  end

  describe "create/1" do
    test "creates a new model with proper folder structure", %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir,
          max_seq_len: 64,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      # Check model struct
      assert %MachineLearning.Transformer.Model{} = model
      assert model.folder == @test_model_dir
      assert model.config["vocab_size"] == 100
      assert model.config["embed_dim"] == 64

      # Check folder structure
      assert File.dir?(@test_model_dir)
      assert File.exists?(Path.join(@test_model_dir, "config.json"))
      assert File.exists?(Path.join(@test_model_dir, "tokenizer.bert"))
      assert File.dir?(Path.join(@test_model_dir, "params"))
      assert File.dir?(Path.join(@test_model_dir, "training_data"))

      # Check initial parameters saved
      assert File.exists?(Path.join([@test_model_dir, "params", "initial.bin"]))
      assert File.exists?(Path.join([@test_model_dir, "params", "latest.bin"]))
    end

    test "creates model with default parameters", %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir
        )

      # Check defaults
      assert model.config["max_seq_len"] == 256
      assert model.config["embed_dim"] == 256
      assert model.config["num_heads"] == 8
      assert model.config["num_layers"] == 4
      assert model.config["ff_dim"] == 1024
    end
  end

  describe "add_training_data/3" do
    setup %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir,
          vocab_size: 100,
          max_seq_len: 64,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      dummy_sequences = create_dummy_sequences()

      {:ok, model: model, sequences: dummy_sequences}
    end

    test "adds training data from token sequences", %{sequences: sequences} do
      :ok =
        Transformer.add_training_data(
          @test_model_dir,
          "test_dataset",
          token_sequences: sequences,
          batch_size: 2,
          seq_len: 16
        )

      # Check dataset folder created
      dataset_dir = Path.join([@test_model_dir, "training_data", "test_dataset"])
      assert File.dir?(dataset_dir)
      assert File.exists?(Path.join(dataset_dir, "metadata.json"))
      assert File.exists?(Path.join(dataset_dir, "batches.bin"))

      # Check metadata
      metadata = File.read!(Path.join(dataset_dir, "metadata.json")) |> Jason.decode!()
      assert metadata["batch_size"] == 2
      assert metadata["seq_len"] == 16
      assert metadata["num_sequences"] == length(sequences)
      assert is_integer(metadata["num_batches"])
    end

    test "adds training data using model struct", %{model: model, sequences: sequences} do
      :ok =
        Transformer.add_training_data(
          model,
          "test_dataset_2",
          token_sequences: sequences,
          batch_size: 4,
          seq_len: 16
        )

      dataset_dir = Path.join([@test_model_dir, "training_data", "test_dataset_2"])
      assert File.dir?(dataset_dir)
    end
  end

  describe "load_training_data/2" do
    setup %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir,
          vocab_size: 100,
          max_seq_len: 64,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      sequences = create_dummy_sequences()

      Transformer.add_training_data(
        @test_model_dir,
        "test_dataset",
        token_sequences: sequences,
        batch_size: 2,
        seq_len: 16
      )

      {:ok, model: model}
    end

    test "loads training data successfully" do
      {train_data, metadata} =
        Transformer.load_training_data(
          @test_model_dir,
          "test_dataset"
        )

      # Check metadata
      assert metadata["batch_size"] == 2
      assert metadata["seq_len"] == 16

      # Check data is enumerable
      batches = Enum.to_list(train_data)
      assert length(batches) > 0

      # Check batch structure
      first_batch = hd(batches)
      assert Map.has_key?(first_batch, :input_ids)
      assert Map.has_key?(first_batch, :labels)
      assert Map.has_key?(first_batch, :attention_mask)

      # Check tensor shapes
      assert {2, 16} = Nx.shape(first_batch.input_ids)
      assert {2, 16} = Nx.shape(first_batch.labels)
      assert {2, 16} = Nx.shape(first_batch.attention_mask)
    end

    test "raises error for non-existent dataset" do
      assert_raise RuntimeError, ~r/Training data 'nonexistent' not found/, fn ->
        Transformer.load_training_data(@test_model_dir, "nonexistent")
      end
    end
  end

  describe "train/3" do
    setup %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir,
          vocab_size: 100,
          max_seq_len: 64,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      sequences = create_dummy_sequences()

      Transformer.add_training_data(
        @test_model_dir,
        "test_dataset",
        token_sequences: sequences,
        batch_size: 2,
        seq_len: 16
      )

      {:ok, model: model}
    end

    test "trains model and saves epoch checkpoints" do
      trained_model =
        Transformer.train(
          @test_model_dir,
          "test_dataset",
          epochs: 2,
          learning_rate: 0.001,
          params_version: "initial"
        )

      # Check model struct returned
      assert %MachineLearning.Transformer.Model{} = trained_model

      # Check epoch checkpoints created
      assert File.exists?(Path.join([@test_model_dir, "params", "epoch_001.bin"]))
      assert File.exists?(Path.join([@test_model_dir, "params", "epoch_002.bin"]))
      assert File.exists?(Path.join([@test_model_dir, "params", "latest.bin"]))
    end

    test "can train from model struct", %{model: model} do
      trained_model =
        Transformer.train(
          model,
          "test_dataset",
          epochs: 1,
          learning_rate: 0.001
        )

      assert %MachineLearning.Transformer.Model{} = trained_model
      assert File.exists?(Path.join([@test_model_dir, "params", "epoch_001.bin"]))
    end
  end

  describe "load/2" do
    setup %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir,
          vocab_size: 100,
          max_seq_len: 64,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      sequences = create_dummy_sequences()

      Transformer.add_training_data(
        @test_model_dir,
        "test_dataset",
        token_sequences: sequences,
        batch_size: 2,
        seq_len: 16
      )

      Transformer.train(
        @test_model_dir,
        "test_dataset",
        epochs: 2,
        learning_rate: 0.001
      )

      {:ok, model: model}
    end

    test "loads model with latest parameters" do
      loaded_model = Transformer.load(@test_model_dir)

      assert %MachineLearning.Transformer.Model{} = loaded_model
      assert loaded_model.folder == @test_model_dir
      assert loaded_model.config["vocab_size"] == 100
    end

    test "loads model with specific parameter version" do
      model_epoch_1 = Transformer.load(@test_model_dir, params_version: "epoch_001")
      assert %MachineLearning.Transformer.Model{} = model_epoch_1

      model_initial = Transformer.load(@test_model_dir, params_version: "initial")
      assert %MachineLearning.Transformer.Model{} = model_initial
    end

    test "raises error for non-existent model directory" do
      assert_raise RuntimeError, ~r/Model directory not found/, fn ->
        Transformer.load("nonexistent_model")
      end
    end

    test "raises error for non-existent parameter version" do
      assert_raise RuntimeError, ~r/Parameters version 'nonexistent' not found/, fn ->
        Transformer.load(@test_model_dir, params_version: "nonexistent")
      end
    end
  end

  describe "predict/3" do
    setup %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir,
          vocab_size: 100,
          max_seq_len: 64,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      {:ok, model: model}
    end

    test "generates text from model path" do
      text =
        Transformer.predict(
          @test_model_dir,
          "the quick",
          max_length: 20,
          temperature: 1.0
        )

      assert is_binary(text)
      assert String.length(text) > 0
    end

    test "generates text from model struct", %{model: model} do
      text =
        Transformer.predict(
          model,
          "machine learning",
          max_length: 15
        )

      assert is_binary(text)
    end
  end

  describe "utility functions" do
    setup %{tokens: tokens} do
      model =
        Transformer.create(
          tokens: tokens,
          save_dir: @test_model_dir,
          vocab_size: 100,
          max_seq_len: 64,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      sequences = create_dummy_sequences()

      Transformer.add_training_data(
        @test_model_dir,
        "dataset_1",
        token_sequences: sequences,
        batch_size: 2,
        seq_len: 16
      )

      Transformer.add_training_data(
        @test_model_dir,
        "dataset_2",
        token_sequences: sequences,
        batch_size: 4,
        seq_len: 16
      )

      Transformer.train(
        @test_model_dir,
        "dataset_1",
        epochs: 2,
        learning_rate: 0.001
      )

      {:ok, model: model}
    end

    test "list_params/1 returns all parameter versions" do
      params = Transformer.list_params(@test_model_dir)

      assert is_list(params)
      assert "initial" in params
      assert "latest" in params
      assert "epoch_001" in params
      assert "epoch_002" in params
    end

    test "list_training_data/1 returns all datasets" do
      datasets = Transformer.list_training_data(@test_model_dir)

      assert is_list(datasets)
      assert "dataset_1" in datasets
      assert "dataset_2" in datasets
      assert length(datasets) == 2
    end

    test "info/1 returns model information" do
      info = Transformer.info(@test_model_dir)

      assert is_map(info)
      assert info["vocab_size"] == 100
      assert info["embed_dim"] == 64
      assert info["num_heads"] == 4
      assert info["num_layers"] == 2
      assert is_list(info["available_params"])
      assert is_list(info["available_datasets"])
      assert length(info["available_datasets"]) == 2
    end
  end

  # Helper functions

  defp create_dummy_tokens do
    # Create a simple vocabulary using MachineLearning.BytePairEncoding.Token
    alias MachineLearning.BytePairEncoding.Token

    # Create special tokens
    special_tokens = [
      Token.new("[PAD]"),
      Token.new("[UNK]"),
      Token.new("[BOS]"),
      Token.new("[EOS]")
    ]

    word_tokens = [
      Token.new("the"),
      Token.new("quick"),
      Token.new("brown"),
      Token.new("fox"),
      Token.new("jumps"),
      Token.new("over"),
      Token.new("lazy"),
      Token.new("dog"),
      Token.new("in"),
      Token.new("a"),
      Token.new("machine"),
      Token.new("learning"),
      Token.new("is"),
      Token.new("fun"),
      Token.new("and"),
      Token.new("powerful"),
      Token.new("transformer"),
      Token.new("model"),
      Token.new("training"),
      Token.new("data"),
      Token.new("neural"),
      Token.new("network")
    ]

    # Pad to 100 tokens
    padding_tokens =
      Enum.map(26..99, fn i ->
        Token.new("token_#{i}")
      end)

    special_tokens ++ word_tokens ++ padding_tokens
  end

  defp create_dummy_sequences do
    [
      # "the quick brown fox jumps over the lazy dog"
      [2, 4, 5, 6, 7, 8, 9, 4, 10, 11, 3],
      # "machine learning is fun and powerful"
      [2, 14, 15, 16, 17, 18, 19, 3],
      # "transformer model training is fun"
      [2, 20, 21, 22, 16, 17, 3],
      # "neural network learning is powerful"
      [2, 24, 25, 15, 16, 19, 3],
      # Repeat to have more training data
      [2, 4, 5, 6, 7, 8, 9, 4, 10, 11, 3],
      [2, 14, 15, 16, 17, 18, 19, 3],
      [2, 20, 21, 22, 16, 17, 3],
      [2, 24, 25, 15, 16, 19, 3]
    ]
  end
end
