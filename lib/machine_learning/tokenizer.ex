defmodule MachineLearning.Tokenizer do
  @moduledoc """
  Tokenizer module for converting text to/from token IDs using BPE vocabulary.

  This module bridges the gap between the BytePairEncoding module and the
  Transformer model by managing the vocabulary and token ID mappings.

  ## Special Tokens

  The tokenizer uses four special tokens (when `add_special_tokens: true`):

  - `<PAD>` (ID: 0) - Padding token used to fill sequences to a uniform length when batching
  - `<UNK>` (ID: 1) - Unknown token used for out-of-vocabulary words that don't exist in the trained vocabulary
  - `<BOS>` (ID: 2) - Beginning of sequence token added at the start of text sequences
  - `<EOS>` (ID: 3) - End of sequence token added at the end of text sequences
  """

  alias MachineLearning.BytePairEncoding.Token

  defstruct [:vocab, :token_to_id, :id_to_token, :vocab_size, :vocab_map]

  @type vocab_map :: %{
          String.t() => %{
            is_token?: boolean(),
            children: vocab_map()
          }
        }

  @type t :: %__MODULE__{
          vocab: list(Token.t()),
          token_to_id: map(),
          id_to_token: map(),
          vocab_size: integer(),
          vocab_map: vocab_map()
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
      vocab_size: length(full_vocab),
      vocab_map: vocab_map_from_vocab(full_vocab)
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

    token_ids =
      text
      |> String.graphemes()
      |> tokenize(tokenizer)
      |> map(tokenizer)

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

  @doc """
  Optimized tokenization using greedy longest-match algorithm.

  ## Parameters

  - `graphemes`: List of graphemes (characters) to tokenize
  - `tokenizer`: The tokenizer struct containing the vocab_map

  ## Returns

  List of Token structs and/or string graphemes for unrecognized characters

  ## Examples

      iex> graphemes = String.graphemes("hello")
      iex> MachineLearning.TokenizerOptimized.tokenize(graphemes, tokenizer)
      [%Token{value: "hello"}]
  """
  @spec tokenize(list(String.t()), MachineLearning.Tokenizer.t()) :: list(Token.t() | String.t())
  def tokenize(graphemes, %MachineLearning.Tokenizer{vocab_map: vocab_map}) do
    tokenize_greedy(graphemes, vocab_map, [])
    |> Enum.reverse()
  end

  @doc """
  Maps tokens to their corresponding IDs.
  ## Parameters
  - `tokens`: List of Token structs or string graphemes
  - `tokenizer`: The tokenizer struct containing the token_to_id mapping
  ## Returns
  List of integer token IDs
  ## Examples
      iex> tokens = [%Token{value: "hello"}, %Token{value: "world"}]
      iex> MachineLearning.Tokenizer.map(tokens, tokenizer)
      [42, 127]
  """
  @spec map(list(Token.t() | String.t()), MachineLearning.Tokenizer.t()) :: list(integer())
  def map(tokens, %MachineLearning.Tokenizer{token_to_id: token_to_id}) do
    Enum.map(tokens, fn
      %Token{value: value} -> Map.get(token_to_id, value, unk_id())
      grapheme when is_binary(grapheme) -> Map.get(token_to_id, grapheme, unk_id())
    end)
  end

  # Greedy longest-match tokenization
  defp tokenize_greedy([], _vocab_map, acc), do: acc

  defp tokenize_greedy(graphemes, vocab_map, acc) do
    case find_longest_token(graphemes, vocab_map) do
      {token, remaining} when token != "" ->
        # Found a token, emit it and continue with remaining graphemes
        tokenize_greedy(remaining, vocab_map, [token | acc])

      {_, [grapheme | remaining]} ->
        # No token found, emit single character and continue
        tokenize_greedy(remaining, vocab_map, [grapheme | acc])

      {_, []} ->
        # Edge case: empty remaining list
        acc
    end
  end

  # Find the longest possible token starting from the beginning of graphemes
  defp find_longest_token(graphemes, vocab_map) do
    find_longest_recursive(graphemes, vocab_map, "", graphemes, 0, 0)
  end

  # Recursively find the longest token match
  # current_chars: characters we're currently checking
  # best_token: longest valid token found so far
  # best_length: length of the best token found
  defp find_longest_recursive(
         [],
         _vocab_map,
         best_token,
         original_graphemes,
         best_length,
         _current_length
       ) do
    if best_length > 0 do
      {best_token, Enum.drop(original_graphemes, best_length)}
    else
      {"", original_graphemes}
    end
  end

  defp find_longest_recursive(
         [char | rest],
         current_map,
         best_token,
         original_graphemes,
         best_length,
         current_length
       ) do
    case Map.get(current_map, char) do
      nil ->
        # No further matches possible, return best found so far
        if best_length > 0 do
          {best_token, Enum.drop(original_graphemes, best_length)}
        else
          {"", original_graphemes}
        end

      %{is_token?: true, children: children} ->
        # This is a valid token, update best and continue looking for longer ones
        current_token = Enum.take(original_graphemes, current_length + 1) |> Enum.join()
        new_best_length = current_length + 1

        find_longest_recursive(
          rest,
          children,
          current_token,
          original_graphemes,
          new_best_length,
          current_length + 1
        )

      %{is_token?: false, children: children} ->
        # Partial match, continue building but don't update best token
        find_longest_recursive(
          rest,
          children,
          best_token,
          original_graphemes,
          best_length,
          current_length + 1
        )
    end
  end

  # Private corpus verification functions

  @spec collect_corpus_files_for_verification(Path.t(), list(String.t()), integer() | nil) ::
          {:ok, list(Path.t())} | {:error, term()}
  defp collect_corpus_files_for_verification(corpus_directory, extensions, max_files) do
    if File.exists?(corpus_directory) do
      files =
        corpus_directory
        |> File.ls!()
        |> Enum.map(&Path.join(corpus_directory, &1))
        |> Enum.flat_map(fn path ->
          if File.dir?(path) do
            collect_files_recursively(path, extensions)
          else
            if Path.extname(path) in extensions, do: [path], else: []
          end
        end)
        |> then(fn files ->
          if max_files, do: Enum.take(files, max_files), else: files
        end)

      {:ok, files}
    else
      {:error, "Corpus directory does not exist: #{corpus_directory}"}
    end
  end

  @spec collect_files_recursively(Path.t(), list(String.t())) :: list(Path.t())
  defp collect_files_recursively(directory, extensions) do
    try do
      directory
      |> File.ls!()
      |> Enum.flat_map(fn filename ->
        full_path = Path.join(directory, filename)

        cond do
          File.dir?(full_path) ->
            collect_files_recursively(full_path, extensions)

          Path.extname(full_path) in extensions ->
            [full_path]

          true ->
            []
        end
      end)
    rescue
      _ -> []
    end
  end

  @spec analyze_corpus_with_tokenizer(t(), list(Path.t()), integer()) ::
          {:ok, map()} | {:error, term()}
  defp analyze_corpus_with_tokenizer(tokenizer, files, sample_size) do
    try do
      # Initialize tracking state
      all_vocab_tokens = MapSet.new(tokenizer.vocab, fn token -> token.value end)
      used_tokens = MapSet.new()
      total_characters = 0
      total_tokens = 0
      unk_count = 0
      unk_sequences = []

      # Process each file and accumulate statistics
      {used_tokens, total_characters, total_tokens, unk_count, unk_sequences} =
        Enum.reduce(
          files,
          {used_tokens, total_characters, total_tokens, unk_count, unk_sequences},
          fn file_path, {used_acc, chars_acc, tokens_acc, unk_acc, unk_seqs_acc} ->
            case File.read(file_path) do
              {:ok, content} ->
                char_count = String.length(content)
                token_ids = encode(tokenizer, content)
                token_count = length(token_ids)

                # Count UNK tokens and sample sequences that became UNK
                {file_unk_count, file_unk_sequences} =
                  analyze_unk_tokens(tokenizer, content, token_ids, sample_size)

                # Track which tokens were used
                file_used_tokens =
                  token_ids
                  |> Enum.map(fn id -> id_to_token(tokenizer, id) end)
                  |> Enum.reject(&is_nil/1)
                  |> MapSet.new()

                {
                  MapSet.union(used_acc, file_used_tokens),
                  chars_acc + char_count,
                  tokens_acc + token_count,
                  unk_acc + file_unk_count,
                  unk_seqs_acc ++
                    Enum.take(file_unk_sequences, max(0, sample_size - length(unk_seqs_acc)))
                }

              {:error, _reason} ->
                # Skip files that can't be read
                {used_acc, chars_acc, tokens_acc, unk_acc, unk_seqs_acc}
            end
          end
        )

      # Calculate final statistics
      unused_tokens = MapSet.difference(all_vocab_tokens, used_tokens)
      used_count = MapSet.size(used_tokens)
      unused_count = MapSet.size(unused_tokens)
      vocab_size = MapSet.size(all_vocab_tokens)

      coverage_percentage =
        if total_tokens > 0 do
          (total_tokens - unk_count) / total_tokens * 100.0
        else
          0.0
        end

      vocab_utilization_percentage =
        if vocab_size > 0 do
          used_count / vocab_size * 100.0
        else
          0.0
        end

      analysis = %{
        total_files: length(files),
        total_characters: total_characters,
        total_tokens: total_tokens,
        used_vocab_tokens: used_count,
        unused_vocab_tokens: unused_count,
        unk_token_count: unk_count,
        coverage_percentage: Float.round(coverage_percentage, 2),
        vocab_utilization_percentage: Float.round(vocab_utilization_percentage, 2),
        used_tokens: used_tokens,
        unused_tokens: unused_tokens,
        sample_unk_sequences: Enum.take(unk_sequences, sample_size)
      }

      {:ok, analysis}
    rescue
      e -> {:error, "Analysis failed: #{Exception.message(e)}"}
    end
  end

  @spec analyze_unk_tokens(t(), String.t(), list(integer()), integer()) ::
          {integer(), list(String.t())}
  defp analyze_unk_tokens(tokenizer, content, token_ids, sample_size) do
    unk_id = unk_id()

    unk_indices =
      token_ids
      |> Enum.with_index()
      |> Enum.filter(fn {id, _idx} -> id == unk_id end)
      |> Enum.map(&elem(&1, 1))

    unk_count = length(unk_indices)

    # Sample some sequences that became UNK for analysis
    graphemes = String.graphemes(content)
    tokenized_parts = tokenize(graphemes, tokenizer)

    unk_sequences =
      tokenized_parts
      |> Enum.with_index()
      |> Enum.filter(fn {token_or_grapheme, _idx} ->
        case token_or_grapheme do
          grapheme when is_binary(grapheme) ->
            # This was a single character that didn't match any token
            Map.get(tokenizer.token_to_id, grapheme, unk_id) == unk_id

          _ ->
            false
        end
      end)
      |> Enum.take(sample_size)
      |> Enum.map(fn {grapheme, _idx} -> grapheme end)

    {unk_count, unk_sequences}
  end

  ## Private helper functions

  defp vocab_map_from_vocab(vocab) do
    vocab
    |> Enum.map(fn token -> token.value end)
    |> Enum.reduce(%{}, fn token, acc -> add_token_to_map(acc, token) end)
  end

  @doc false
  @spec add_token_to_map(vocab_map(), String.t()) :: vocab_map()
  defp add_token_to_map(map, token) do
    add_token_recursive(map, String.graphemes(token), [])
  end

  # Recursively add a token to the trie structure
  # When we've consumed all characters, mark this node as a token
  defp add_token_recursive(map, [], _path) do
    map
  end

  defp add_token_recursive(map, [char | rest], path) do
    current_key = char
    is_final = rest == []

    # Get or create entry for this character
    entry = Map.get(map, current_key, %{is_token?: false, children: %{}})

    # If this is the final character, mark it as a token
    entry =
      if is_final do
        Map.put(entry, :is_token?, true)
      else
        entry
      end

    # Recursively add the rest of the characters to children
    children = Map.get(entry, :children, %{})
    updated_children = add_token_recursive(children, rest, path ++ [char])

    # Update the entry with the new children
    updated_entry = Map.put(entry, :children, updated_children)
    Map.put(map, current_key, updated_entry)
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
          # ASCII printable
          # Whitespace
          # Valid Unicode (before surrogates)
          # Valid Unicode (after surrogates)
          (char >= 32 and char <= 126) or
            char in [?\n, ?\r, ?\t, ?\s] or
            (char >= 160 and char < 0xD800) or
            (char > 0xDFFF and char <= 0x10FFFF)

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
  Verifies tokenizer performance against a corpus of files.

  Analyzes how well the tokenizer handles the given corpus by computing
  statistics on token usage and coverage.

  ## Parameters

  - `tokenizer`: The tokenizer struct
  - `corpus_directory`: Directory containing text files to analyze
  - `opts`: Options
    - `:max_files` - Maximum number of files to process (default: nil, process all)
    - `:file_extensions` - List of file extensions to process (default: [".txt"])
    - `:sample_size` - Number of UNK sequences to sample for analysis (default: 10)

  ## Returns

  {:ok, %{
    total_files: integer(),
    total_characters: integer(),
    total_tokens: integer(),
    used_vocab_tokens: integer(),
    unused_vocab_tokens: integer(),
    unk_token_count: integer(),
    coverage_percentage: float(),
    vocab_utilization_percentage: float(),
    used_tokens: MapSet.t(),
    unused_tokens: MapSet.t(),
    sample_unk_sequences: list()
  }} | {:error, reason}

  ## Examples

      iex> {:ok, stats} = MachineLearning.Tokenizer.verify(tokenizer, "/path/to/corpus")
      iex> stats.coverage_percentage
      95.5
  """
  @spec verify(t(), Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def verify(%__MODULE__{} = tokenizer, corpus_directory, opts \\ []) do
    max_files = Keyword.get(opts, :max_files)
    file_extensions = Keyword.get(opts, :file_extensions, [".txt"])
    sample_size = Keyword.get(opts, :sample_size, 10)

    with {:ok, files} <-
           collect_corpus_files_for_verification(corpus_directory, file_extensions, max_files),
         {:ok, analysis} <- analyze_corpus_with_tokenizer(tokenizer, files, sample_size) do
      {:ok, analysis}
    else
      {:error, reason} -> {:error, reason}
    end
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

  # Padding token - fills sequences to uniform length for batching
  defp pad_id, do: 0

  # Unknown token - represents out-of-vocabulary words
  defp unk_id, do: 1

  # Beginning of sequence token - marks the start of text
  defp bos_id, do: 2

  # End of sequence token - marks the end of text
  defp eos_id, do: 3

  defp is_special_token?(token) do
    token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
  end
end
