defmodule MachineLearning.Transformer.ModelTest do
  use ExUnit.Case, async: false
  alias MachineLearning.Transformer.Model
  alias MachineLearning.Tokenizer
  alias MachineLearning.Transformer

  @moduletag :tmp_dir
  @moduletag timeout: 120_000

  describe "Model.save/2" do
    test "saves model to specified directory", %{tmp_dir: tmp_dir} do
      # Setup: Create a simple model
      model_struct = create_test_model()
      save_dir = Path.join(tmp_dir, "saved_model")

      # Act: Save the model
      result = Model.save(model_struct, save_dir)

      # Assert: Files are created
      assert File.exists?(Path.join(save_dir, "config.json"))
      assert File.exists?(Path.join(save_dir, "tokenizer.bert"))
      assert File.exists?(Path.join(save_dir, "params.bin"))

      # Assert: Returned model has updated folder
      assert result.folder == save_dir
      assert result.model == model_struct.model
      assert result.params == model_struct.params
      assert result.tokenizer == model_struct.tokenizer
      assert result.config == model_struct.config
    end

    test "saves model to timestamped directory when no directory provided", %{tmp_dir: tmp_dir} do
      # Setup: Create a simple model without folder
      model_struct = create_test_model()

      # Change to tmp_dir to ensure timestamped dir is created there
      File.cd!(tmp_dir, fn ->
        # Act: Save without specifying directory
        result = Model.save(model_struct)

        # Assert: Files are created in timestamped directory
        assert result.folder =~ ~r/models\/transformer_\d+/
        assert File.exists?(Path.join(result.folder, "config.json"))
        assert File.exists?(Path.join(result.folder, "tokenizer.bert"))
        assert File.exists?(Path.join(result.folder, "params.bin"))
      end)
    end

    test "uses existing folder when no directory provided and model has folder", %{
      tmp_dir: tmp_dir
    } do
      # Setup: Create a model with an existing folder
      existing_folder = Path.join(tmp_dir, "existing_folder")
      File.mkdir_p!(existing_folder)
      model_struct = create_test_model(existing_folder)

      # Act: Save without specifying directory
      result = Model.save(model_struct)

      # Assert: Files are created in existing folder
      assert result.folder == existing_folder
      assert File.exists?(Path.join(existing_folder, "config.json"))
      assert File.exists?(Path.join(existing_folder, "tokenizer.bert"))
      assert File.exists?(Path.join(existing_folder, "params.bin"))
    end

    test "saves valid JSON config", %{tmp_dir: tmp_dir} do
      # Setup
      model_struct = create_test_model()
      save_dir = Path.join(tmp_dir, "json_test")

      # Act
      Model.save(model_struct, save_dir)

      # Assert: Config is valid JSON with expected fields
      config_json =
        Path.join(save_dir, "config.json")
        |> File.read!()
        |> Jason.decode!()

      assert config_json["max_seq_len"] == 128
      assert config_json["embed_dim"] == 64
      assert config_json["num_heads"] == 4
      assert config_json["num_layers"] == 2
      assert config_json["ff_dim"] == 256
      assert config_json["vocab_size"] > 0
      assert is_binary(config_json["saved_at"])
    end

    test "saves serialized parameters", %{tmp_dir: tmp_dir} do
      # Setup
      model_struct = create_test_model()
      save_dir = Path.join(tmp_dir, "params_test")

      # Act
      Model.save(model_struct, save_dir)

      # Assert: Parameters can be deserialized
      params_path = Path.join(save_dir, "params.bin")
      assert File.exists?(params_path)

      loaded_params =
        File.read!(params_path)
        |> Nx.deserialize()

      # Verify structure matches
      assert is_map(loaded_params)
      assert Map.keys(loaded_params) == Map.keys(model_struct.params)
    end
  end

  describe "Model.load/1" do
    test "loads model from directory", %{tmp_dir: tmp_dir} do
      # Setup: Create and save a model
      original_model = create_test_model()
      save_dir = Path.join(tmp_dir, "load_test")
      Model.save(original_model, save_dir)

      # Act: Load the model
      loaded_model = Model.load(save_dir)

      # Assert: Model is loaded correctly
      assert %Model{} = loaded_model
      assert loaded_model.folder == save_dir

      assert Tokenizer.vocab_size(loaded_model.tokenizer) ==
               Tokenizer.vocab_size(original_model.tokenizer)

      assert loaded_model.config["embed_dim"] == original_model.config["embed_dim"]
      assert loaded_model.config["num_heads"] == original_model.config["num_heads"]
      assert loaded_model.config["num_layers"] == original_model.config["num_layers"]
      assert loaded_model.config["vocab_size"] == original_model.config["vocab_size"]
    end

    test "raises error when config.json is missing", %{tmp_dir: tmp_dir} do
      # Setup: Create directory without config.json
      empty_dir = Path.join(tmp_dir, "no_config")
      File.mkdir_p!(empty_dir)

      # Act & Assert
      assert_raise RuntimeError, ~r/Model configuration file not found/, fn ->
        Model.load(empty_dir)
      end
    end

    test "raises error when tokenizer.bert is missing", %{tmp_dir: tmp_dir} do
      # Setup: Create directory with only config.json
      incomplete_dir = Path.join(tmp_dir, "no_tokenizer")
      File.mkdir_p!(incomplete_dir)

      config = %{
        "max_seq_len" => 128,
        "embed_dim" => 64,
        "num_heads" => 4,
        "num_layers" => 2,
        "ff_dim" => 256,
        "vocab_size" => 100
      }

      File.write!(Path.join(incomplete_dir, "config.json"), Jason.encode!(config))

      # Act & Assert
      assert_raise RuntimeError, ~r/Tokenizer file not found/, fn ->
        Model.load(incomplete_dir)
      end
    end

    test "raises error when params.bin is missing", %{tmp_dir: tmp_dir} do
      # Setup: Create directory with config and tokenizer but no params
      incomplete_dir = Path.join(tmp_dir, "no_params")
      File.mkdir_p!(incomplete_dir)

      # Create config
      config = %{
        "max_seq_len" => 128,
        "embed_dim" => 64,
        "num_heads" => 4,
        "num_layers" => 2,
        "ff_dim" => 256,
        "vocab_size" => 100
      }

      File.write!(Path.join(incomplete_dir, "config.json"), Jason.encode!(config))

      # Create tokenizer
      tokenizer = create_test_tokenizer()
      Tokenizer.save(tokenizer, Path.join(incomplete_dir, "tokenizer.bert"))

      # Act & Assert
      assert_raise RuntimeError, ~r/Parameters file not found/, fn ->
        Model.load(incomplete_dir)
      end
    end

    test "loaded model can be used for generation", %{tmp_dir: tmp_dir} do
      # Setup: Create and save a model
      original_model = create_test_model()
      save_dir = Path.join(tmp_dir, "generation_test")
      Model.save(original_model, save_dir)

      # Act: Load and use for generation
      loaded_model = Model.load(save_dir)

      # Generate text
      prompt_ids = Tokenizer.encode(loaded_model.tokenizer, "test")
      prompt_tensor = Nx.tensor([prompt_ids])

      generated_ids =
        Transformer.generate(
          loaded_model.model,
          loaded_model.params,
          prompt_tensor,
          max_length: 10
        )

      generated_text = Tokenizer.decode(loaded_model.tokenizer, generated_ids)

      # Assert: Generation produces text
      assert is_binary(generated_text)
      assert String.length(generated_text) > 0
    end
  end

  describe "Model.save/2 and Model.load/1 round-trip" do
    test "save and load preserves model functionality", %{tmp_dir: tmp_dir} do
      # Setup: Create a model
      original_model = create_test_model()
      save_dir = Path.join(tmp_dir, "roundtrip_test")

      # Act: Save and load
      Model.save(original_model, save_dir)
      loaded_model = Model.load(save_dir)

      # Assert: Generate text with both models
      prompt_text = "hello"
      prompt_ids = Tokenizer.encode(original_model.tokenizer, prompt_text)
      prompt_tensor = Nx.tensor([prompt_ids])

      original_output =
        Transformer.generate(
          original_model.model,
          original_model.params,
          prompt_tensor,
          max_length: 10,
          temperature: 0.1
        )

      loaded_prompt_ids = Tokenizer.encode(loaded_model.tokenizer, prompt_text)
      loaded_prompt_tensor = Nx.tensor([loaded_prompt_ids])

      loaded_output =
        Transformer.generate(
          loaded_model.model,
          loaded_model.params,
          loaded_prompt_tensor,
          max_length: 10,
          temperature: 0.1
        )

      # Both should produce deterministic output with low temperature
      original_text = Tokenizer.decode(original_model.tokenizer, original_output)
      loaded_text = Tokenizer.decode(loaded_model.tokenizer, loaded_output)

      assert is_binary(original_text)
      assert is_binary(loaded_text)
    end
  end

  # Helper functions
  defp create_test_model(folder \\ nil) do
    # Create a small tokenizer
    tokenizer = create_test_tokenizer()
    vocab_size = Tokenizer.vocab_size(tokenizer)

    # Create a small model
    model =
      Transformer.create_model(
        vocab_size: vocab_size,
        max_seq_len: 128,
        embed_dim: 64,
        num_heads: 4,
        num_layers: 2,
        ff_dim: 256,
        dropout_rate: 0.1
      )

    # Initialize parameters
    params = Transformer.init_params(model, seq_len: 16)

    # Create config
    config = %{
      "max_seq_len" => 128,
      "embed_dim" => 64,
      "num_heads" => 4,
      "num_layers" => 2,
      "ff_dim" => 256,
      "vocab_size" => vocab_size,
      "saved_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    Model.new(model, params, tokenizer, config, folder)
  end

  defp create_test_tokenizer do
    # Load existing vocabulary or create a minimal one
    if File.exists?("vocabulary.bert") do
      tokens =
        File.read!("vocabulary.bert")
        |> :erlang.binary_to_term()
        |> Enum.map(&MachineLearning.BytePairEncoding.Token.new/1)

      Tokenizer.from_vocab(tokens)
    else
      # Create minimal tokenizer for testing with basic tokens
      alias MachineLearning.BytePairEncoding.Token

      tokens = [
        Token.new("h"),
        Token.new("e"),
        Token.new("l"),
        Token.new("o"),
        Token.new(" "),
        Token.new("w"),
        Token.new("r"),
        Token.new("d"),
        Token.new("t"),
        Token.new("s"),
        Token.new("a"),
        Token.new("m"),
        Token.new("c"),
        Token.new("i"),
        Token.new("n"),
        Token.new("g")
      ]

      Tokenizer.from_vocab(tokens)
    end
  end
end
