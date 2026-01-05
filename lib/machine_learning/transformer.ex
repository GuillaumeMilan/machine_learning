defmodule MachineLearning.Transformer do
  @moduledoc """
  Main API for creating, training, and using transformer language models.

  This module provides a high-level interface for the complete transformer model lifecycle:
  - Creating new models with initial parameters
  - Adding and managing training datasets
  - Training with automatic checkpoint saving
  - Loading models for inference or continued training
  - Generating text predictions

  ## Folder Structure

  Each model is stored in a directory with the following structure:

      models/my_model/
      ├── config.json              # Model architecture configuration
      ├── tokenizer.bert           # Vocabulary and tokenizer
      ├── params/                  # Versioned model parameters
      │   ├── initial.bin          # Initial parameters
      │   ├── epoch_001.bin        # After epoch 1
      │   ├── epoch_002.bin        # After epoch 2
      │   └── latest.bin           # Most recent parameters
      └── training_data/           # Prepared training datasets
          ├── dataset_v1/
          │   ├── metadata.json    # Dataset configuration
          │   └── batches.bin      # Serialized training batches
          └── dataset_v2/
              ├── metadata.json
              └── batches.bin

  ## Example Workflow

      # 1. Create a new model
      tokens = File.read!("vocabulary.bert") |> :erlang.binary_to_term()
      model = MachineLearning.Transformer.create(
        tokens: tokens,
        save_dir: "models/my_model",
        embed_dim: 256,
        num_heads: 8,
        num_layers: 4,
        ff_dim: 1024
      )

      # 2. Add training data
      MachineLearning.Transformer.add_training_data(
        "models/my_model",
        "dataset_v1",
        corpus_dir: "./tmp/corpus",
        batch_size: 32,
        seq_len: 128
      )

      # 3. Train the model
      MachineLearning.Transformer.train(
        "models/my_model",
        "dataset_v1",
        epochs: 10,
        learning_rate: 0.001
      )

      # 4. Generate text
      text = MachineLearning.Transformer.predict(
        "models/my_model",
        "The quick brown",
        max_length: 50
      )
  """

  alias MachineLearning.Tokenizer
  alias MachineLearning.Transformer.Backend
  alias MachineLearning.Transformer.Model
  alias MachineLearning.Transformer.Corpus

  @doc """
  Creates a new transformer model with initial parameters.

  ## Parameters

  - `opts`: Keyword list of options
    - `:tokens` - List of BPE tokens for creating the tokenizer (required)
    - `:save_dir` - Directory path to save the model (required)
    - `:max_seq_len` - Maximum sequence length (default: 256)
    - `:embed_dim` - Embedding dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_layers` - Number of transformer layers (default: 4)
    - `:ff_dim` - Feed-forward dimension (default: 1024)

  ## Examples

      # Load tokens from vocabulary file
      tokens = File.read!("vocabulary.bert") |> :erlang.binary_to_term()

      model = MachineLearning.Transformer.create(
        tokens: tokens,
        save_dir: "models/small",
        embed_dim: 128,
        max_seq_len: 128,
        num_heads: 8,
        num_layers: 2,
        ff_dim: 512
      )
  """
  @spec create(keyword()) :: Model.t()
  def create(opts) do
    # Extract required parameters
    tokens = Keyword.fetch!(opts, :tokens)
    save_dir = Keyword.fetch!(opts, :save_dir)

    # Extract optional model parameters with defaults
    max_seq_len = Keyword.get(opts, :max_seq_len, 256)
    embed_dim = Keyword.get(opts, :embed_dim, 256)
    num_heads = Keyword.get(opts, :num_heads, 8)
    num_layers = Keyword.get(opts, :num_layers, 4)
    ff_dim = Keyword.get(opts, :ff_dim, 1024)

    IO.puts("Creating new transformer model at: #{save_dir}")

    # Create directory structure
    File.mkdir_p!(save_dir)
    File.mkdir_p!(Path.join(save_dir, "params"))
    File.mkdir_p!(Path.join(save_dir, "training_data"))

    # Create tokenizer from tokens
    tokenizer = Tokenizer.from_vocab(tokens)

    # Create model configuration
    config = %{
      "vocab_size" => tokenizer.vocab_size,
      "max_seq_len" => max_seq_len,
      "embed_dim" => embed_dim,
      "num_heads" => num_heads,
      "num_layers" => num_layers,
      "ff_dim" => ff_dim,
      "created_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    # Save config
    config_path = Path.join(save_dir, "config.json")
    File.write!(config_path, Jason.encode!(config, pretty: true))
    IO.puts("✓ Saved model configuration")

    # Save tokenizer
    tokenizer_path = Path.join(save_dir, "tokenizer.bert")
    Tokenizer.save(tokenizer, tokenizer_path)
    IO.puts("✓ Saved tokenizer")

    # Create Axon model
    model =
      Backend.create_model(
        vocab_size: tokenizer.vocab_size,
        max_seq_len: max_seq_len,
        embed_dim: embed_dim,
        num_heads: num_heads,
        num_layers: num_layers,
        ff_dim: ff_dim,
        dropout_rate: 0.1
      )

    # Initialize parameters
    IO.puts("Initializing model parameters...")
    params = Backend.init_params(model, seq_len: max_seq_len)
    IO.puts("✓ Parameters initialized")

    # Save initial parameters
    initial_params_path = Path.join([save_dir, "params", "initial.bin"])
    File.write!(initial_params_path, Nx.serialize(params))
    IO.puts("✓ Saved initial parameters")

    # Also save as latest
    latest_params_path = Path.join([save_dir, "params", "latest.bin"])
    File.write!(latest_params_path, Nx.serialize(params))
    IO.puts("✓ Saved latest parameters")

    IO.puts("\n✅ Model created successfully at: #{save_dir}")

    Model.new(model, params, tokenizer, config, save_dir)
  end

  @doc """
  Adds training data to a model's training_data folder.

  ## Parameters

  - `model_or_path`: Either a Model struct or path to model directory
  - `dataset_name`: Name/identifier for this dataset
  - `opts`: Options
    - `:corpus_dir` - Load texts from corpus directory
    - `:token_sequences` - Pre-tokenized sequences (alternative to corpus_dir)
    - `:batch_size` - Batch size for training (default: 32)
    - `:seq_len` - Sequence length (default: 128)
    - `:sample_size` - Limit number of texts from corpus (optional)
    - `:shuffle` - Whether to shuffle data (default: true)

  ## Examples

      # From corpus directory
      MachineLearning.Transformer.add_training_data(
        "models/small",
        "dataset_elixir",
        corpus_dir: "/tmp/elixir",
        batch_size: 32,
        seq_len: 128
      )

      # From pre-tokenized sequences
      MachineLearning.Transformer.add_training_data(
        model,
        "dataset_v2",
        token_sequences: [[1, 2, 3], [4, 5, 6]],
        batch_size: 16,
        seq_len: 64
      )
  """
  @spec add_training_data(Model.t() | Path.t(), String.t(), keyword()) :: :ok
  def add_training_data(model_or_path, dataset_name, opts) do
    # Load model if path provided
    {model_dir, tokenizer} =
      case model_or_path do
        %Model{folder: folder, tokenizer: tok} ->
          {folder, tok}

        path when is_binary(path) ->
          model = load(path)
          {model.folder, model.tokenizer}
      end

    IO.puts("\nAdding training data '#{dataset_name}' to model at: #{model_dir}")

    # Extract options
    batch_size = Keyword.get(opts, :batch_size, 32)
    seq_len = Keyword.get(opts, :seq_len, 128)
    shuffle = Keyword.get(opts, :shuffle, true)

    # Get token sequences
    token_sequences =
      if corpus_dir = Keyword.get(opts, :corpus_dir) do
        IO.puts("Loading texts from corpus: #{corpus_dir}")

        unless File.dir?(corpus_dir) do
          raise "Corpus directory not found: #{corpus_dir}"
        end

        # Load texts from corpus
        sample_size = Keyword.get(opts, :sample_size)

        texts =
          if sample_size do
            Corpus.load_texts(corpus_dir, max_files: sample_size)
          else
            Corpus.load_texts(corpus_dir)
          end

        IO.puts("Loaded #{length(texts)} texts from corpus")

        # Tokenize texts
        IO.puts("Tokenizing texts...")

        texts
        |> Enum.map(fn text ->
          Tokenizer.encode(tokenizer, text, add_special_tokens: true)
        end)
      else
        # Use provided token sequences
        Keyword.fetch!(opts, :token_sequences)
      end

    IO.puts("Processing #{length(token_sequences)} token sequences...")

    # Prepare batched training data
    train_data =
      Backend.prepare_training_data(
        token_sequences,
        batch_size: batch_size,
        seq_len: seq_len,
        shuffle: shuffle
      )

    # Materialize the stream into a list
    IO.puts("Materializing batches...")
    batches = Enum.to_list(train_data)
    num_batches = length(batches)
    IO.puts("Created #{num_batches} batches")

    # Create dataset directory
    dataset_dir = Path.join([model_dir, "training_data", dataset_name])
    File.mkdir_p!(dataset_dir)

    # Save metadata
    metadata = %{
      "batch_size" => batch_size,
      "seq_len" => seq_len,
      "num_batches" => num_batches,
      "num_sequences" => length(token_sequences),
      "shuffle" => shuffle,
      "created_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    metadata_path = Path.join(dataset_dir, "metadata.json")
    File.write!(metadata_path, Jason.encode!(metadata, pretty: true))
    IO.puts("✓ Saved metadata")

    # Serialize and save batches
    IO.puts("Serializing batches...")
    batches_path = Path.join(dataset_dir, "batches.bin")

    # Convert batches to a serializable format
    serialized_batches =
      Enum.map(batches, fn batch ->
        %{
          "input_ids" => Nx.serialize(batch.input_ids),
          "labels" => Nx.serialize(batch.labels),
          "attention_mask" => Nx.serialize(batch.attention_mask)
        }
      end)

    File.write!(batches_path, :erlang.term_to_binary(serialized_batches))
    IO.puts("✓ Saved batches")

    IO.puts("\n✅ Training data '#{dataset_name}' added successfully")
    IO.puts("   Location: #{dataset_dir}")
    IO.puts("   Batches: #{num_batches}")
    IO.puts("   Batch size: #{batch_size}")
    IO.puts("   Sequence length: #{seq_len}")

    :ok
  end

  @doc """
  Loads training data from a model's training_data folder.

  ## Parameters

  - `model_or_path`: Either a Model struct or path to model directory
  - `dataset_name`: Name of the dataset to load

  ## Returns

  A tuple of `{train_data_stream, metadata}` where train_data_stream is an
  enumerable of batches and metadata contains dataset configuration.

  ## Examples

      {train_data, metadata} = MachineLearning.Transformer.load_training_data(
        "models/my_model",
        "dataset_v1"
      )
  """
  @spec load_training_data(Model.t() | Path.t(), String.t()) :: {Enumerable.t(), map()}
  def load_training_data(model_or_path, dataset_name) do
    # Get model directory
    model_dir =
      case model_or_path do
        %Model{folder: folder} -> folder
        path when is_binary(path) -> path
      end

    dataset_dir = Path.join([model_dir, "training_data", dataset_name])

    unless File.dir?(dataset_dir) do
      raise "Training data '#{dataset_name}' not found at: #{dataset_dir}"
    end

    # Load metadata
    metadata_path = Path.join(dataset_dir, "metadata.json")
    metadata = File.read!(metadata_path) |> Jason.decode!()

    # Load batches
    batches_path = Path.join(dataset_dir, "batches.bin")
    serialized_batches = File.read!(batches_path) |> :erlang.binary_to_term()

    # Deserialize batches lazily
    train_data =
      Stream.map(serialized_batches, fn batch ->
        %{
          input_ids: Nx.deserialize(batch["input_ids"]),
          labels: Nx.deserialize(batch["labels"]),
          attention_mask: Nx.deserialize(batch["attention_mask"])
        }
      end)

    {train_data, metadata}
  end

  @doc """
  Trains a transformer model on a specific dataset.

  Automatically saves parameters after each epoch to params/epoch_XXX.bin
  and updates params/latest.bin.

  ## Parameters

  - `model_or_path`: Either a Model struct or path to model directory
  - `dataset_name`: Name of the training dataset to use
  - `opts`: Training options
    - `:epochs` - Number of epochs to train (default: 10)
    - `:learning_rate` - Learning rate (default: 0.001)
    - `:params_version` - Which parameter version to start from (default: "latest")
    - `:save_every_epoch` - Save parameters after each epoch (default: true)
    - `:optimizer` - Optimizer to use (default: :adamw)

  ## Examples

      # Train from scratch
      model = MachineLearning.Transformer.train(
        "models/small",
        "dataset_elixir",
        epochs: 10,
        params_version: "latest"
      )

      # Continue training
      model = MachineLearning.Transformer.train(
        "models/my_model",
        "dataset_v1",
        epochs: 10,
        params_version: "latest"
      )
  """
  @spec train(Model.t() | Path.t(), String.t(), keyword()) :: Model.t()
  def train(model_or_path, dataset_name, opts \\ []) do
    # Load model if path provided
    model =
      case model_or_path do
        %Model{} = m ->
          m

        path when is_binary(path) ->
          params_version = Keyword.get(opts, :params_version, "latest")
          load(path, params_version: params_version)
      end

    # Extract training options
    epochs = Keyword.get(opts, :epochs, 10)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    save_every_epoch = Keyword.get(opts, :save_every_epoch, true)
    optimizer = Keyword.get(opts, :optimizer, :adamw)

    IO.puts("\n🚀 Starting training")
    IO.puts("   Model: #{model.folder}")
    IO.puts("   Dataset: #{dataset_name}")
    IO.puts("   Epochs: #{epochs}")
    IO.puts("   Learning rate: #{learning_rate}")

    # Load training data
    {train_data, metadata} = load_training_data(model, dataset_name)
    IO.puts("   Batches: #{metadata["num_batches"]}")
    IO.puts("   Batch size: #{metadata["batch_size"]}")
    IO.puts("   Sequence length: #{metadata["seq_len"]}\n")

    # Train epoch by epoch to save checkpoints
    final_params =
      Enum.reduce(1..epochs, model.params, fn epoch, current_params ->
        IO.puts("\n📊 Epoch #{epoch}/#{epochs}")

        # Train for one epoch
        updated_params =
          Backend.train(
            model.model,
            current_params,
            train_data,
            epochs: 1,
            learning_rate: learning_rate,
            optimizer: optimizer
          )

        # Save checkpoint if enabled
        if save_every_epoch do
          epoch_str = String.pad_leading("#{epoch}", 3, "0")
          params_path = Path.join([model.folder, "params", "epoch_#{epoch_str}.bin"])
          File.write!(params_path, Nx.serialize(updated_params))
          IO.puts("✓ Saved checkpoint: epoch_#{epoch_str}.bin")
        end

        # Update latest
        latest_path = Path.join([model.folder, "params", "latest.bin"])
        File.write!(latest_path, Nx.serialize(updated_params))

        updated_params
      end)

    IO.puts("\n✅ Training complete!")

    # Return updated model
    %{model | params: final_params}
  end

  @doc """
  Loads a transformer model from disk.

  ## Parameters

  - `model_dir`: Path to the model directory
  - `opts`: Options
    - `:params_version` - Which parameter version to load (default: "latest")

  ## Examples

      # Load with latest parameters
      model = MachineLearning.Transformer.load("models/my_model")

      # Load with specific epoch
      model = MachineLearning.Transformer.load(
        "models/my_model",
        params_version: "epoch_005"
      )

      # Load initial parameters
      model = MachineLearning.Transformer.load(
        "models/my_model",
        params_version: "initial"
      )
  """
  @spec load(Path.t(), keyword()) :: Model.t()
  def load(model_dir, opts \\ []) do
    params_version = Keyword.get(opts, :params_version, "latest")

    unless File.dir?(model_dir) do
      raise "Model directory not found: #{model_dir}"
    end

    IO.puts("Loading model from: #{model_dir}")
    IO.puts("Parameters version: #{params_version}")

    # Load config
    config_path = Path.join(model_dir, "config.json")
    config = File.read!(config_path) |> Jason.decode!()

    # Load tokenizer
    tokenizer_path = Path.join(model_dir, "tokenizer.bert")
    tokenizer = Tokenizer.load(tokenizer_path)

    # Load parameters
    params_path = Path.join([model_dir, "params", "#{params_version}.bin"])

    unless File.exists?(params_path) do
      raise "Parameters version '#{params_version}' not found at: #{params_path}"
    end

    params = File.read!(params_path) |> Nx.deserialize()

    # Create model
    model =
      Backend.create_model(
        vocab_size: config["vocab_size"],
        max_seq_len: config["max_seq_len"],
        embed_dim: config["embed_dim"],
        num_heads: config["num_heads"],
        num_layers: config["num_layers"],
        ff_dim: config["ff_dim"],
        dropout_rate: 0.1
      )

    IO.puts("✓ Model loaded successfully\n")

    Model.new(model, params, tokenizer, config, model_dir)
  end

  @doc """
  Generates text using a trained model.

  ## Parameters

  - `model_or_path`: Either a Model struct or path to model directory
  - `prompt_text`: Text prompt to start generation
  - `opts`: Generation options (same as Backend.generate/4)

  ## Examples

      text = MachineLearning.Transformer.predict(
        "models/my_model",
        "The quick brown",
        max_length: 50,
        temperature: 0.9
      )
  """
  @spec predict(Model.t() | Path.t(), String.t(), keyword()) :: String.t()
  def predict(model_or_path, prompt_text, opts \\ []) do
    # Load model if path provided
    model =
      case model_or_path do
        %Model{} = m -> m
        path when is_binary(path) -> load(path)
      end

    # Encode prompt
    prompt_ids = Tokenizer.encode(model.tokenizer, prompt_text)
    prompt_tensor = Nx.tensor([prompt_ids])

    # Generate
    generated_ids =
      Backend.generate(
        model.model,
        model.params,
        prompt_tensor,
        opts
      )

    # Decode
    Tokenizer.decode(model.tokenizer, generated_ids)
  end

  @doc """
  Lists all available parameter versions for a model.

  ## Examples

      MachineLearning.Transformer.list_params("models/my_model")
      # => ["initial", "epoch_001", "epoch_002", "latest"]
  """
  @spec list_params(Model.t() | Path.t()) :: list(String.t())
  def list_params(model_or_path) do
    model_dir =
      case model_or_path do
        %Model{folder: folder} -> folder
        path when is_binary(path) -> path
      end

    params_dir = Path.join(model_dir, "params")

    unless File.dir?(params_dir) do
      raise "Parameters directory not found: #{params_dir}"
    end

    File.ls!(params_dir)
    |> Enum.filter(&String.ends_with?(&1, ".bin"))
    |> Enum.map(&Path.rootname/1)
    |> Enum.sort()
  end

  @doc """
  Lists all available training datasets for a model.

  ## Examples

      MachineLearning.Transformer.list_training_data("models/my_model")
      # => ["dataset_v1", "dataset_v2"]
  """
  @spec list_training_data(Model.t() | Path.t()) :: list(String.t())
  def list_training_data(model_or_path) do
    model_dir =
      case model_or_path do
        %Model{folder: folder} -> folder
        path when is_binary(path) -> path
      end

    training_data_dir = Path.join(model_dir, "training_data")

    unless File.dir?(training_data_dir) do
      []
    else
      File.ls!(training_data_dir)
      |> Enum.filter(fn name ->
        File.dir?(Path.join(training_data_dir, name))
      end)
      |> Enum.sort()
    end
  end

  @doc """
  Returns information about a model.

  ## Examples

      MachineLearning.Transformer.info("models/my_model")
      # => %{
      #   "vocab_size" => 5000,
      #   "embed_dim" => 256,
      #   "num_heads" => 8,
      #   ...
      # }
  """
  @spec info(Model.t() | Path.t()) :: map()
  def info(model_or_path) do
    model_dir =
      case model_or_path do
        %Model{folder: folder, config: config} ->
          Map.put(config, "folder", folder)

        path when is_binary(path) ->
          config_path = Path.join(path, "config.json")
          config = File.read!(config_path) |> Jason.decode!()
          Map.put(config, "folder", path)
      end

    # Add available params and datasets
    model_dir
    |> Map.put("available_params", list_params(model_dir["folder"]))
    |> Map.put("available_datasets", list_training_data(model_dir["folder"]))
  end
end
