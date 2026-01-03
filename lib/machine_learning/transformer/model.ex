defmodule MachineLearning.Transformer.Model do
  @moduledoc """
  Represents a complete transformer model with all its components.

  This struct encapsulates the Axon model, trained parameters, tokenizer,
  model configuration, and the directory from which it was loaded.
  """

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

      iex> model_struct = MachineLearning.Transformer.Model.new(model, params, tokenizer, config, "/path/to/model")
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
end
