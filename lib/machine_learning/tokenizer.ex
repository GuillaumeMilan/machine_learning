defmodule MachineLearning.Tokenizer do
  @moduledoc """
  Tokenizer module for converting text to/from token IDs using BPE vocabulary.

  This module bridges the gap between the BytePairEncoding module and the
  Transformer model by managing the vocabulary and token ID mappings.
  """

  alias MachineLearning.BytePairEncoding
  alias MachineLearning.BytePairEncoding.Token

  defstruct [:vocab, :token_to_id, :id_to_token, :vocab_size]

  @type t :: %__MODULE__{
          vocab: list(Token.t()),
          token_to_id: map(),
          id_to_token: map(),
          vocab_size: integer()
        }

  @doc """
  Creates a tokenizer from a BPE vocabulary.

  ## Parameters

  - `vocab`: List of BPE tokens (from BytePairEncoding.compress/3)
  - `opts`: Options
    - `:add_special_tokens` - Add special tokens like PAD, UNK, etc. (default: true)

  ## Examples

      iex> vocab = MachineLearning.BytePairEncoding.compress(corpus_dir, 5000)
      iex> tokenizer = MachineLearning.Tokenizer.from_vocab(vocab)
  """
  @spec from_vocab(list(Token.t()), keyword()) :: t()
  def from_vocab(vocab, opts \\ []) do
    add_special = Keyword.get(opts, :add_special_tokens, true)

    # Add special tokens at the beginning
    full_vocab =
      if add_special do
        special_tokens = [
          Token.new("<PAD>"),
          Token.new("<UNK>"),
          Token.new("<BOS>"),
          Token.new("<EOS>")
        ]

        special_tokens ++ vocab
      else
        vocab
      end

    # Create bidirectional mappings
    token_to_id =
      full_vocab
      |> Enum.with_index()
      |> Map.new(fn {token, idx} -> {token.value, idx} end)

    id_to_token =
      full_vocab
      |> Enum.with_index()
      |> Map.new(fn {token, idx} -> {idx, token.value} end)

    %__MODULE__{
      vocab: full_vocab,
      token_to_id: token_to_id,
      id_to_token: id_to_token,
      vocab_size: length(full_vocab)
    }
  end

  @doc """
  Loads a tokenizer from a saved vocabulary file.

  ## Parameters

  - `path`: Path to the vocabulary file

  ## Examples

      iex> tokenizer = MachineLearning.Tokenizer.load("vocabulary.bert")
  """
  @spec load(Path.t()) :: t()
  def load(path) do
    vocab =
      path
      |> File.read!()
      |> :erlang.binary_to_term()

    from_vocab(vocab)
  end

  @doc """
  Saves the tokenizer vocabulary to a file.

  ## Parameters

  - `tokenizer`: The tokenizer struct
  - `path`: Path where to save the vocabulary

  ## Examples

      iex> MachineLearning.Tokenizer.save(tokenizer, "vocabulary.bert")
  """
  @spec save(t(), Path.t()) :: :ok
  def save(%__MODULE__{vocab: vocab}, path) do
    # Save only the non-special tokens (exclude the first 4 if they are special tokens)
    non_special_vocab =
      case vocab do
        [
          %Token{value: "<PAD>"},
          %Token{value: "<UNK>"},
          %Token{value: "<BOS>"},
          %Token{value: "<EOS>"} | rest
        ] ->
          rest

        _ ->
          vocab
      end

    File.write!(path, :erlang.term_to_binary(non_special_vocab))
  end

  @doc """
  Encodes text into token IDs.

  ## Parameters

  - `tokenizer`: The tokenizer struct
  - `text`: Text to encode (string or list of strings)
  - `opts`: Encoding options
    - `:add_special_tokens` - Add BOS/EOS tokens (default: false)
    - `:max_length` - Maximum sequence length (default: nil)
    - `:padding` - Pad to max_length (default: false)

  ## Examples

      iex> token_ids = MachineLearning.Tokenizer.encode(tokenizer, "Hello world")
      [42, 127, 89, ...]

      iex> token_ids = MachineLearning.Tokenizer.encode(
      ...>   tokenizer,
      ...>   "Hello world",
      ...>   add_special_tokens: true,
      ...>   max_length: 128,
      ...>   padding: true
      ...> )
  """
  @spec encode(t(), String.t() | list(String.t()), keyword()) :: list(integer())
  def encode(%__MODULE__{} = tokenizer, text, opts \\ []) when is_binary(text) do
    add_special = Keyword.get(opts, :add_special_tokens, false)
    max_length = Keyword.get(opts, :max_length)
    padding = Keyword.get(opts, :padding, false)

    # Tokenize text into BPE tokens
    graphemes = String.graphemes(text)
    tokens = BytePairEncoding.tokenize(graphemes, tokenizer.vocab)

    # Convert tokens to IDs
    token_ids =
      tokens
      |> Enum.map(fn
        %Token{value: value} -> Map.get(tokenizer.token_to_id, value, unk_id())
        grapheme when is_binary(grapheme) -> Map.get(tokenizer.token_to_id, grapheme, unk_id())
      end)

    # Add special tokens if requested
    token_ids =
      if add_special do
        [bos_id() | token_ids] ++ [eos_id()]
      else
        token_ids
      end

    # Apply max_length and padding
    token_ids =
      cond do
        max_length && length(token_ids) > max_length ->
          Enum.take(token_ids, max_length)

        max_length && padding ->
          token_ids ++ List.duplicate(pad_id(), max_length - length(token_ids))

        true ->
          token_ids
      end

    token_ids
  end

  @doc """
  Encodes multiple texts into a batch of token IDs.

  ## Parameters

  - `tokenizer`: The tokenizer struct
  - `texts`: List of texts to encode
  - `opts`: Encoding options (same as encode/3)

  ## Examples

      iex> batch_ids = MachineLearning.Tokenizer.encode_batch(
      ...>   tokenizer,
      ...>   ["Hello world", "Goodbye world"],
      ...>   max_length: 128,
      ...>   padding: true
      ...> )
  """
  @spec encode_batch(t(), list(String.t()), keyword()) :: list(list(integer()))
  def encode_batch(%__MODULE__{} = tokenizer, texts, opts \\ []) do
    Enum.map(texts, fn text -> encode(tokenizer, text, opts) end)
  end

  @doc """
  Decodes token IDs back into text.

  ## Parameters

  - `tokenizer`: The tokenizer struct
  - `token_ids`: List of token IDs or Nx tensor
  - `opts`: Decoding options
    - `:skip_special_tokens` - Skip special tokens in output (default: true)

  ## Examples

      iex> text = MachineLearning.Tokenizer.decode(tokenizer, [42, 127, 89])
      "Hello world"
  """
  @spec decode(t(), list(integer()) | Nx.Tensor.t(), keyword()) :: String.t()
  def decode(tokenizer, token_ids, opts \\ [])

  def decode(tokenizer, %Nx.Tensor{} = token_ids_tensor, opts) do
    token_ids = Nx.to_flat_list(token_ids_tensor)
    decode(tokenizer, token_ids, opts)
  end

  def decode(%__MODULE__{id_to_token: id_to_token}, token_ids, opts) when is_list(token_ids) do
    skip_special = Keyword.get(opts, :skip_special_tokens, true)

    token_ids
    |> Enum.map(fn id -> Map.get(id_to_token, id, "<UNK>") end)
    |> then(fn tokens ->
      if skip_special do
        Enum.reject(tokens, &is_special_token?/1)
      else
        tokens
      end
    end)
    |> Enum.join("")
    |> sanitize_unicode()
  end

  # Sanitize the decoded text to ensure valid UTF-8 and printable characters
  defp sanitize_unicode(text) do
    # First, try to ensure it's valid UTF-8
    sanitized =
      case :unicode.characters_to_binary(text, :utf8, :utf8) do
        binary when is_binary(binary) -> binary
        {:error, _, _} -> scrub_invalid_utf8(text)
        {:incomplete, _, _} -> scrub_invalid_utf8(text)
      end

    # Then filter out problematic control characters but keep printable Unicode
    sanitized
    |> String.graphemes()
    |> Enum.filter(fn grapheme ->
      case String.to_charlist(grapheme) do
        [char] ->
          # Keep printable ASCII, common whitespace, and valid extended Unicode
          (char >= 32 and char <= 126) or  # ASCII printable
          char in [?\n, ?\r, ?\t, ?\s] or  # Whitespace
          (char >= 160 and char < 0xD800) or  # Valid Unicode (before surrogates)
          (char > 0xDFFF and char <= 0x10FFFF)  # Valid Unicode (after surrogates)

        _ ->
          # Multi-codepoint graphemes (like emojis) - keep them
          true
      end
    end)
    |> Enum.join()
  end

  # Replace invalid UTF-8 sequences with the replacement character
  defp scrub_invalid_utf8(binary) do
    binary
    |> String.codepoints()
    |> Enum.map(fn codepoint ->
      if String.valid?(codepoint), do: codepoint, else: "?"
    end)
    |> Enum.join()
  end

  @doc """
  Decodes a batch of token ID sequences.

  ## Parameters

  - `tokenizer`: The tokenizer struct
  - `batch_ids`: List of token ID lists
  - `opts`: Decoding options (same as decode/3)

  ## Examples

      iex> texts = MachineLearning.Tokenizer.decode_batch(tokenizer, batch_ids)
      ["Hello world", "Goodbye world"]
  """
  @spec decode_batch(t(), list(list(integer())), keyword()) :: list(String.t())
  def decode_batch(%__MODULE__{} = tokenizer, batch_ids, opts \\ []) do
    Enum.map(batch_ids, fn ids -> decode(tokenizer, ids, opts) end)
  end

  @doc """
  Returns the vocabulary size.

  ## Examples

      iex> MachineLearning.Tokenizer.vocab_size(tokenizer)
      5000
  """
  @spec vocab_size(t()) :: integer()
  def vocab_size(%__MODULE__{vocab_size: size}), do: size

  @doc """
  Returns the token ID for a specific token string.

  ## Examples

      iex> MachineLearning.Tokenizer.token_to_id(tokenizer, "hello")
      42
  """
  @spec token_to_id(t(), String.t()) :: integer() | nil
  def token_to_id(%__MODULE__{token_to_id: mapping}, token) do
    Map.get(mapping, token)
  end

  @doc """
  Returns the token string for a specific token ID.

  ## Examples

      iex> MachineLearning.Tokenizer.id_to_token(tokenizer, 42)
      "hello"
  """
  @spec id_to_token(t(), integer()) :: String.t() | nil
  def id_to_token(%__MODULE__{id_to_token: mapping}, id) do
    Map.get(mapping, id)
  end

  # Special token IDs (when add_special_tokens is true)
  defp pad_id, do: 0
  defp unk_id, do: 1
  defp bos_id, do: 2
  defp eos_id, do: 3

  defp is_special_token?(token) do
    token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
  end
end
