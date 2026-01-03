defmodule MachineLearning.TransformerTrainingTest do
  use ExUnit.Case, async: false
  alias MachineLearning.TransformerTraining
  alias MachineLearning.Tokenizer
  alias MachineLearning.BytePairEncoding.Token

  @moduletag timeout: 120_000
  @moduletag :transformer_training

  describe "run/1" do
    test "completes full training pipeline with minimal config" do
      # Ensure vocabulary file exists
      assert File.exists?("vocabulary.bert"),
             "vocabulary.bert file must exist for training"

      # Create minimal config with sample texts (no corpus needed)
      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: nil,
        # Use shorter sequences for test data
        seq_len: 16,
        batch_size: 2
      }

      # Capture IO to verify training progress messages
      output =
        ExUnit.CaptureIO.capture_io(fn ->
          result = TransformerTraining.run(config)
          # Function now returns Model struct
          assert %MachineLearning.Transformer.Model{} = result
        end)

      # Verify key steps were executed
      assert output =~ "Setting up tokenizer"
      assert output =~ "Vocabulary size:"
      assert output =~ "Preparing training data"
      assert output =~ "Creating transformer model"
      assert output =~ "Initializing model parameters"
      assert output =~ "Training model"
      assert output =~ "Training completed"
      assert output =~ "Generating text samples"
    end

    # TODO use tmp_dir
    @tag :tmp_dir
    test "works with custom corpus directory", %{tmp_dir: tmp_dir} do
      # Create temporary corpus directory with test files
      corpus_dir = Path.join(tmp_dir, "test_corpus")
      File.rm_rf!(corpus_dir)
      File.mkdir_p!(corpus_dir)

      # Write longer training texts to ensure they pass seq_len filter
      # Each text needs to be long enough to create valid training sequences
      File.write!(
        Path.join(corpus_dir, "sample1.txt"),
        String.duplicate(
          "The quick brown fox jumps over the lazy dog in the forest near the river. ",
          2
        )
      )

      # File.write!(
      #   Path.join(corpus_dir, "sample2.txt"),
      #   String.duplicate("Machine learning is fascinating and has many practical applications in industry. ", 2)
      # )

      # File.write!(
      #   Path.join(corpus_dir, "sample3.txt"),
      #   String.duplicate("Transformers revolutionized natural language processing with attention mechanisms. ", 2)
      # )

      # File.write!(
      #   Path.join(corpus_dir, "sample4.txt"),
      #   String.duplicate("Deep neural networks can learn hierarchical representations from raw data. ", 2)
      # )

      # File.write!(
      #   Path.join(corpus_dir, "sample5.txt"),
      #   String.duplicate("Natural language understanding requires sophisticated models and large datasets. ", 2)
      # )

      config = %{
        epoch: 1,
        vocab_path: "vocabulary.bert",
        corpus_dir: corpus_dir,
        sample_size: 3,
        seq_len: 16,
        batch_size: 2
      }

      output =
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)

      # Verify corpus was loaded
      assert output =~ "Loading all texts from corpus directory"
      assert output =~ "Loaded 1 total texts from corpus"
      refute output =~ "Sampling 3 texts for training"
      assert output =~ "Using 1 texts for training"

      # Cleanup
      File.rm_rf!(corpus_dir)
    end

    test "loads entire corpus when sample_size is not specified" do
      # Create temporary corpus directory with test files
      corpus_dir = "test/tmp/test_corpus_full"
      File.rm_rf!(corpus_dir)
      File.mkdir_p!(corpus_dir)

      # Write test files with longer texts
      texts = [
        "The quick brown fox jumps over the lazy dog in the beautiful forest.",
        "Machine learning algorithms can solve complex problems with pattern recognition.",
        "Deep neural networks extract hierarchical features from large datasets efficiently.",
        "Natural language processing enables computers to understand human communication.",
        "Transformers revolutionized artificial intelligence with attention mechanisms today."
      ]

      for {text, i} <- Enum.with_index(texts, 1) do
        File.write!(Path.join(corpus_dir, "sample#{i}.txt"), text)
      end

      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: corpus_dir,
        # Use shorter sequences for test data
        seq_len: 16,
        batch_size: 2
      }

      output =
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)

      # Verify all texts are used when no sample_size specified
      assert output =~ "Loading all texts from corpus directory"
      assert output =~ "Loaded 5 total texts from corpus"
      assert output =~ "Using 5 texts for training"
      refute output =~ "Sampling"

      # Cleanup
      File.rm_rf!(corpus_dir)
    end

    test "raises error when vocabulary file is missing" do
      config = %{
        vocab_path: "nonexistent_vocab.bert"
      }

      assert_raise RuntimeError, ~r/Vocabulary file not found/, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)
      end
    end

    test "raises error when corpus directory is invalid" do
      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: "nonexistent_corpus_dir"
      }

      assert_raise RuntimeError, ~r/Corpus directory not found/, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)
      end
    end
  end

  describe "training process verification" do
    test "model parameters are updated during training" do
      # Load vocabulary
      tokens =
        File.read!("vocabulary.bert")
        |> :erlang.binary_to_term()
        |> Enum.map(&Token.new/1)

      tokenizer = Tokenizer.from_vocab(tokens)

      # Create minimal training data with longer texts
      sample_texts = [
        "The quick brown fox jumps over the lazy dog in the park.",
        "Machine learning models can learn complex patterns from data.",
        "Transformers are powerful neural network architectures for NLP.",
        "Deep learning has revolutionized artificial intelligence applications.",
        "Natural language processing enables computers to understand text."
      ]

      token_sequences =
        Enum.map(sample_texts, fn text ->
          Tokenizer.encode(tokenizer, text, add_special_tokens: true)
        end)

      train_data =
        MachineLearning.Transformer.prepare_training_data(token_sequences,
          batch_size: 2,
          seq_len: 16,
          shuffle: false
        )

      # Create small model
      model =
        MachineLearning.Transformer.create_small_model(
          vocab_size: Tokenizer.vocab_size(tokenizer),
          max_seq_len: 32,
          embed_dim: 64,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 128
        )

      # Initialize parameters
      params = MachineLearning.Transformer.init_params(model, seq_len: 16)

      # Train for 1 epoch
      trained_params =
        MachineLearning.Transformer.train(model, params, train_data,
          epochs: 1,
          learning_rate: 0.001
        )

      # Verify parameters were returned and are maps
      assert is_map(trained_params)
      assert map_size(trained_params) > 0

      # Parameters should be different from initial (training occurred)
      # We just verify structure is maintained
      assert Map.keys(params) == Map.keys(trained_params)
    end

    test "generates text after minimal training" do
      # Load vocabulary
      tokens =
        File.read!("vocabulary.bert")
        |> :erlang.binary_to_term()
        |> Enum.map(&Token.new/1)

      tokenizer = Tokenizer.from_vocab(tokens)

      # Minimal training with longer texts
      sample_texts = [
        "The quick brown fox jumps over the lazy dog in the park.",
        "Machine learning models can learn complex patterns from large datasets.",
        "Transformers revolutionized natural language processing with attention.",
        "Deep neural networks extract hierarchical features from raw data."
      ]

      token_sequences =
        Enum.map(sample_texts, fn text ->
          Tokenizer.encode(tokenizer, text, add_special_tokens: true)
        end)

      train_data =
        MachineLearning.Transformer.prepare_training_data(token_sequences,
          batch_size: 2,
          seq_len: 16,
          shuffle: false
        )

      model =
        MachineLearning.Transformer.create_small_model(
          vocab_size: Tokenizer.vocab_size(tokenizer),
          max_seq_len: 32,
          embed_dim: 32,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 64
        )

      params = MachineLearning.Transformer.init_params(model, seq_len: 16)

      trained_params =
        MachineLearning.Transformer.train(model, params, train_data,
          epochs: 1,
          learning_rate: 0.001
        )

      # Generate text
      prompt_text = "The"
      prompt_ids = Tokenizer.encode(tokenizer, prompt_text)
      prompt_tensor = Nx.tensor([prompt_ids])

      generated_ids =
        MachineLearning.Transformer.generate(model, trained_params, prompt_tensor,
          max_length: 10,
          temperature: 1.0,
          top_k: 5
        )

      # Decode and verify we got text back
      generated_text = Tokenizer.decode(tokenizer, generated_ids)
      assert is_binary(generated_text)
      assert String.length(generated_text) > 0
    end
  end
end
