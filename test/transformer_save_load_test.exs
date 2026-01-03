defmodule MachineLearning.TransformerSaveLoadTest do
  use ExUnit.Case, async: false
  alias MachineLearning.TransformerTraining
  alias MachineLearning.Tokenizer
  alias MachineLearning.Transformer

  @moduletag :tmp_dir
  @moduletag :transformer_training
  @moduletag timeout: 120_000

  describe "save and load model" do
    test "saves and loads model with tokenizer and params", %{tmp_dir: tmp_dir} do
      # Ensure vocabulary file exists
      assert File.exists?("vocabulary.bert"),
             "vocabulary.bert file must exist for training"

      # Train a small model
      save_dir = Path.join(tmp_dir, "test_model")

      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: nil,
        epoch: 1,
        save_dir: save_dir,
        # Use shorter sequences for test data
        seq_len: 16,
        batch_size: 2
      }

      # Suppress output during training (may contain unicode issues)
      ExUnit.CaptureIO.capture_io(:stderr, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)
      end)

      # Verify all files were created
      assert File.exists?(Path.join(save_dir, "config.json"))
      assert File.exists?(Path.join(save_dir, "tokenizer.bert"))
      assert File.exists?(Path.join(save_dir, "params.bin"))

      # Verify config.json content
      config_json =
        Path.join(save_dir, "config.json")
        |> File.read!()
        |> Jason.decode!()

      # Check saved configuration matches new defaults or passed config
      # New default
      assert config_json["max_seq_len"] == 256
      # New default
      assert config_json["embed_dim"] == 256
      # New default
      assert config_json["num_heads"] == 8
      # New default
      assert config_json["num_layers"] == 4
      # New default
      assert config_json["ff_dim"] == 1024
      assert is_integer(config_json["vocab_size"])
      assert config_json["vocab_size"] > 0
      assert is_binary(config_json["saved_at"])

      # Load the model
      load_output =
        ExUnit.CaptureIO.capture_io(fn ->
          {model, params, tokenizer} = TransformerTraining.load(save_dir)

          # Verify loaded components
          assert model != nil
          assert params != nil
          assert tokenizer != nil

          # Verify tokenizer is functional
          vocab_size = Tokenizer.vocab_size(tokenizer)
          assert vocab_size == config_json["vocab_size"]

          # Verify model is an Axon model
          assert %Axon{} = model
        end)

      # Verify load messages
      assert load_output =~ "Loading model from #{save_dir}"
      assert load_output =~ "Loaded model configuration"
      assert load_output =~ "Loaded tokenizer with vocabulary size:"
      assert load_output =~ "Loaded trained parameters"
      assert load_output =~ "Created model with saved architecture"
      # New default
      assert load_output =~ "embed_dim=256"
      # New default
      assert load_output =~ "num_heads=8"
      # New default
      assert load_output =~ "num_layers=4"
      assert load_output =~ "Model loaded successfully"
    end

    test "can generate text with loaded model", %{tmp_dir: tmp_dir} do
      # Train a small model
      save_dir = Path.join(tmp_dir, "test_model_generate")

      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: nil,
        epoch: 1,
        save_dir: save_dir,
        seq_len: 16,
        batch_size: 2
      }

      # Suppress output during training (may contain unicode issues)
      ExUnit.CaptureIO.capture_io(:stderr, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)
      end)

      # Load the model and generate text
      {model, params, tokenizer} =
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.load(save_dir)
        end)
        |> then(fn _ -> TransformerTraining.load(save_dir) end)

      # Generate text from a prompt
      prompt_text = "The quick brown"
      prompt_ids = Tokenizer.encode(tokenizer, prompt_text)
      prompt_tensor = Nx.tensor([prompt_ids])

      generated_ids =
        Transformer.generate(model, params, prompt_tensor,
          max_length: 10,
          temperature: 0.8,
          top_k: 5
        )

      generated_text = Tokenizer.decode(tokenizer, generated_ids)

      # Verify generation produces some output
      assert is_binary(generated_text)
      assert String.length(generated_text) > 0
    end

    test "saves with custom directory name", %{tmp_dir: tmp_dir} do
      custom_save_dir = Path.join(tmp_dir, "my_custom_model")

      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: nil,
        epoch: 1,
        save_dir: custom_save_dir,
        seq_len: 16,
        batch_size: 2
      }

      # Suppress output during training (may contain unicode issues)
      ExUnit.CaptureIO.capture_io(:stderr, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)
      end)

      # Verify custom directory was used
      assert File.dir?(custom_save_dir)
      assert File.exists?(Path.join(custom_save_dir, "config.json"))
    end

    test "raises error when loading from non-existent directory", %{tmp_dir: tmp_dir} do
      non_existent = Path.join(tmp_dir, "non_existent_model")

      assert_raise RuntimeError, ~r/Model configuration file not found/, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.load(non_existent)
        end)
      end
    end

    test "raises error when config.json is missing", %{tmp_dir: tmp_dir} do
      incomplete_dir = Path.join(tmp_dir, "incomplete_model")
      File.mkdir_p!(incomplete_dir)

      # Create only tokenizer and params, but no config
      File.write!(Path.join(incomplete_dir, "tokenizer.bert"), "dummy")
      File.write!(Path.join(incomplete_dir, "params.bin"), "dummy")

      assert_raise RuntimeError, ~r/Model configuration file not found/, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.load(incomplete_dir)
        end)
      end
    end

    test "raises error when tokenizer.bert is missing", %{tmp_dir: tmp_dir} do
      incomplete_dir = Path.join(tmp_dir, "incomplete_model_no_tokenizer")
      File.mkdir_p!(incomplete_dir)

      # Create config and params but no tokenizer
      config = %{
        "max_seq_len" => 128,
        "embed_dim" => 128,
        "num_heads" => 4,
        "num_layers" => 2,
        "ff_dim" => 512
      }

      File.write!(
        Path.join(incomplete_dir, "config.json"),
        Jason.encode!(config)
      )

      File.write!(Path.join(incomplete_dir, "params.bin"), "dummy")

      assert_raise RuntimeError, ~r/Tokenizer file not found/, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.load(incomplete_dir)
        end)
      end
    end

    test "raises error when params.bin is missing", %{tmp_dir: tmp_dir} do
      incomplete_dir = Path.join(tmp_dir, "incomplete_model_no_params")
      File.mkdir_p!(incomplete_dir)

      # Create a minimal vocabulary for tokenizer
      tokens = [
        MachineLearning.BytePairEncoding.Token.new("<PAD>"),
        MachineLearning.BytePairEncoding.Token.new("<UNK>"),
        MachineLearning.BytePairEncoding.Token.new("test")
      ]

      tokenizer = Tokenizer.from_vocab(tokens)
      Tokenizer.save(tokenizer, Path.join(incomplete_dir, "tokenizer.bert"))

      # Create config
      config = %{
        "max_seq_len" => 128,
        "embed_dim" => 128,
        "num_heads" => 4,
        "num_layers" => 2,
        "ff_dim" => 512
      }

      File.write!(
        Path.join(incomplete_dir, "config.json"),
        Jason.encode!(config)
      )

      # No params.bin

      assert_raise RuntimeError, ~r/Parameters file not found/, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.load(incomplete_dir)
        end)
      end
    end

    test "config.json is valid JSON with expected fields", %{tmp_dir: tmp_dir} do
      save_dir = Path.join(tmp_dir, "test_config_validation")

      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: nil,
        epoch: 1,
        save_dir: save_dir,
        seq_len: 16,
        batch_size: 2
      }

      # Suppress output during training (may contain unicode issues)
      ExUnit.CaptureIO.capture_io(:stderr, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)
      end)

      config_path = Path.join(save_dir, "config.json")
      assert File.exists?(config_path)

      # Parse and validate JSON structure
      config_json =
        config_path
        |> File.read!()
        |> Jason.decode!()

      # Check all required fields exist
      assert Map.has_key?(config_json, "max_seq_len")
      assert Map.has_key?(config_json, "embed_dim")
      assert Map.has_key?(config_json, "num_heads")
      assert Map.has_key?(config_json, "num_layers")
      assert Map.has_key?(config_json, "ff_dim")
      assert Map.has_key?(config_json, "vocab_size")
      assert Map.has_key?(config_json, "saved_at")

      # Validate field types and values
      assert is_integer(config_json["max_seq_len"])
      assert is_integer(config_json["embed_dim"])
      assert is_integer(config_json["num_heads"])
      assert is_integer(config_json["num_layers"])
      assert is_integer(config_json["ff_dim"])
      assert is_integer(config_json["vocab_size"])
      assert is_binary(config_json["saved_at"])

      # Validate saved_at is a valid ISO8601 timestamp
      assert {:ok, _datetime, _offset} = DateTime.from_iso8601(config_json["saved_at"])
    end

    test "multiple save/load cycles preserve model functionality", %{tmp_dir: tmp_dir} do
      # First training and save
      save_dir_1 = Path.join(tmp_dir, "model_cycle_1")

      config = %{
        vocab_path: "vocabulary.bert",
        corpus_dir: nil,
        epoch: 1,
        save_dir: save_dir_1,
        seq_len: 16,
        batch_size: 2
      }

      # Suppress output during training (may contain unicode issues)
      ExUnit.CaptureIO.capture_io(:stderr, fn ->
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.run(config)
        end)
      end)

      # Load first model
      {model_1, params_1, tokenizer_1} =
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.load(save_dir_1)
        end)
        |> then(fn _ -> TransformerTraining.load(save_dir_1) end)

      # Generate text with first model
      prompt_ids = Tokenizer.encode(tokenizer_1, "Test prompt")
      prompt_tensor = Nx.tensor([prompt_ids])

      _generated_ids =
        ExUnit.CaptureIO.capture_io(fn ->
          Transformer.generate(model_1, params_1, prompt_tensor, max_length: 5)
        end)

      # Load again and verify it still works
      {model_2, _params_2, tokenizer_2} =
        ExUnit.CaptureIO.capture_io(fn ->
          TransformerTraining.load(save_dir_1)
        end)
        |> then(fn _ -> TransformerTraining.load(save_dir_1) end)

      # Both loads should work identically
      assert Tokenizer.vocab_size(tokenizer_1) == Tokenizer.vocab_size(tokenizer_2)
      assert %Axon{} = model_1
      assert %Axon{} = model_2
    end
  end
end
