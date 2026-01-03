defmodule MachineLearning.Transformer.Model do
  @moduledoc """
  Represents a complete transformer model with all its components.

  This struct encapsulates the Axon model, trained parameters, tokenizer,
  model configuration, and the directory from which it was loaded.
  """

  alias MachineLearning.Tokenizer
  alias MachineLearning.Transformer.Backend

  @type t :: %__MODULE__{
          model: Axon.t(),
          params: map(),
          tokenizer: MachineLearning.Tokenizer.t(),
          config: map(),
          folder: String.t() | nil
        }

  defstruct [:model, :params, :tokenizer, :config, :folder]

  @doc """
  Creates a new Model struct.

  ## Parameters

  - `model`: The Axon model
  - `params`: The trained model parameters
  - `tokenizer`: The tokenizer for encoding/decoding text
  - `config`: The model configuration map
  - `folder`: Optional path to the directory where the model is saved (default: nil)

  ## Examples

      iex> model_struct = MachineLearning.Backend.Model.new(model, params, tokenizer, config, "/path/to/model")
  """
  def new(model, params, tokenizer, config, folder \\ nil) do
    %__MODULE__{
      model: model,
      params: params,
      tokenizer: tokenizer,
      config: config,
      folder: folder
    }
  end

  @doc """
  Load a trained model and tokenizer from a directory.

  ## Parameters

  - `model_dir`: Path to the directory containing saved model files

  ## Returns

  A `MachineLearning.Backend.Model` struct containing the loaded model, trained parameters, tokenizer,
  configuration, and the folder path. The model configuration is automatically loaded from the saved `config.json` file.

  ## Examples

      iex> model = MachineLearning.Backend.Model.load("models/transformer_1735862400")
      iex> # Generate text
      iex> prompt_ids = Tokenizer.encode(model.tokenizer, "The quick brown")
      iex> prompt_tensor = Nx.tensor([prompt_ids])
      iex> generated_ids = Backend.generate(model.model, model.params, prompt_tensor, max_length: 20)
      iex> generated_text = Tokenizer.decode(model.tokenizer, generated_ids)
  """
  @spec load(Path.t()) :: t()
  def load(model_dir) do
    # Load model configuration
    model_config_path = Path.join(model_dir, "config.json")

    unless File.exists?(model_config_path) do
      raise "Model configuration file not found: #{model_config_path}"
    end

    model_config =
      File.read!(model_config_path)
      |> Jason.decode!()

    # Load tokenizer
    tokenizer_path = Path.join(model_dir, "tokenizer.bert")

    unless File.exists?(tokenizer_path) do
      raise "Tokenizer file not found: #{tokenizer_path}"
    end

    tokenizer = Tokenizer.load(tokenizer_path)

    # Load parameters
    params_path = Path.join(model_dir, "params.bin")

    unless File.exists?(params_path) do
      raise "Parameters file not found: #{params_path}"
    end

    params =
      File.read!(params_path)
      |> Nx.deserialize()

    # Create model with the saved architecture
    model =
      model_config
      |> model_config_to_model_opts()
      |> Backend.create_model()

    new(model, params, tokenizer, model_config, model_dir)
  end

  @doc """
  Save the model and tokenizer to a directory.

  ## Parameters

  - `model_struct`: A `MachineLearning.Backend.Model` struct
  - `save_dir`: Optional directory path to save the model. If not provided, a timestamped directory will be created.

  ## Returns

  The updated `MachineLearning.Backend.Model` struct with the folder path set to the save directory.

  ## Examples

      iex> model = MachineLearning.Backend.Model.load("models/transformer_1735862400")
      iex> updated_model = MachineLearning.Backend.Model.save(model, "models/my_model")
  """
  @spec save(t(), Path.t() | nil) :: t()
  def save(%__MODULE__{} = model_struct, save_dir \\ nil) do
    # Create save directory with timestamp if not provided
    save_dir =
      save_dir ||
        model_struct.folder ||
        "models/transformer_#{DateTime.utc_now() |> DateTime.to_unix()}"

    # Create directory if it doesn't exist
    File.mkdir_p!(save_dir)

    # Save model configuration
    config_path = Path.join(save_dir, "config.json")
    File.write!(config_path, Jason.encode!(model_struct.config, pretty: true))

    # Save tokenizer
    tokenizer_path = Path.join(save_dir, "tokenizer.bert")
    Tokenizer.save(model_struct.tokenizer, tokenizer_path)

    # Save trained parameters using Nx.serialize
    params_path = Path.join(save_dir, "params.bin")
    serialized_params = Nx.serialize(model_struct.params)
    File.write!(params_path, serialized_params)

    %{model_struct | folder: save_dir}
  end

  # Convert model config map to model options keyword list
  defp model_config_to_model_opts(model_config) do
    embed_dim = Map.fetch!(model_config, "embed_dim")
    num_heads = Map.fetch!(model_config, "num_heads")
    num_layers = Map.fetch!(model_config, "num_layers")
    ff_dim = Map.fetch!(model_config, "ff_dim")
    max_seq_len = Map.fetch!(model_config, "max_seq_len")
    vocab_size = Map.fetch!(model_config, "vocab_size")

    [
      vocab_size: vocab_size,
      max_seq_len: max_seq_len,
      embed_dim: embed_dim,
      num_heads: num_heads,
      num_layers: num_layers,
      ff_dim: ff_dim,
      dropout_rate: 0.1
    ]
  end
end
