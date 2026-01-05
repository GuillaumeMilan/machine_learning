defmodule MachineLearning.Transformer.Corpus do
  @moduledoc """
  Corpus management and evaluation utilities for transformer training.

  This module provides functions to:
  - Add files to a corpus directory
  - Analyze corpus statistics
  - Evaluate corpus quality and suitability for model training
  """

  require Logger

  alias MachineLearning.Transformer.Model
  alias MachineLearning.Tokenizer

  @default_extensions [
    ".ex",
    ".exs",
    ".md",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".py",
    ".java",
    ".c",
    ".cpp",
    ".rb",
    ".go",
    ".rs",
    ".html",
    ".css",
    ".json",
    ".yml",
    ".yaml",
    ".toml"
  ]

  @max_concurrency 50

  @doc """
  Adds files from a source directory to a corpus directory.

  ## Parameters

  - `source_directory`: Path to the directory containing source files
  - `corpus_directory`: Path to the corpus directory where files will be stored
  - `opts`: Optional configuration map
    - `:extensions` - List of file extensions to include (default: common code extensions)

  ## Examples

      iex> MachineLearning.Transformer.Corpus.add_dir(
      ...>   "src/code",
      ...>   "corpus/training",
      ...>   %{extensions: [".ex", ".exs"]}
      ...> )
      :ok
  """
  @spec add_dir(Path.t(), Path.t(), map()) :: :ok
  def add_dir(source_directory, corpus_directory, opts \\ %{}) do
    accepted_extensions = Map.get(opts, :extensions, @default_extensions)
    File.mkdir_p!(corpus_directory)

    list_files(source_directory)
    |> Stream.filter(fn filename -> Path.extname(filename) in accepted_extensions end)
    |> Task.async_stream(
      fn filename ->
        file_content = File.read!(filename)

        reject? =
          file_content
          |> String.split("\n")
          |> Enum.any?(fn line -> String.length(String.trim(line)) >= 1000 end)

        if reject? do
          Logger.warning("Skipping file with very long lines: #{filename}")
          :skip
        else
          content = filename_header(filename) <> File.read!(filename)
          corpus_name = :crypto.hash(:sha256, content) |> Base.encode16() |> String.downcase()
          corpus_file_path = Path.join(corpus_directory, corpus_file_path(corpus_name))
          File.mkdir_p!(Path.dirname(corpus_file_path))
          File.write!(corpus_file_path, content)
        end
      end,
      max_concurrency: @max_concurrency,
      timeout: :infinity
    )
    |> Stream.run()
  end

  @doc """
  Loads all text files from a corpus directory.

  ## Parameters

  - `corpus_directory`: Path to the corpus directory
  - `opts`: Options
    - `:max_files` - Maximum number of files to load (default: nil, loads all)
    - `:min_length` - Minimum text length in characters (default: 10)
    - `:max_length` - Maximum text length in characters (default: nil)
    - `:shuffle` - Whether to shuffle the files (default: false)

  ## Examples

      iex> texts = MachineLearning.Corpus.load_texts("./tmp/corpus")
      iex> is_list(texts)
      true

      iex> texts = MachineLearning.Corpus.load_texts(
      ...>   "./tmp/corpus",
      ...>   max_files: 100,
      ...>   min_length: 50,
      ...>   shuffle: true
      ...> )
  """
  @spec load_texts(Path.t(), keyword()) :: list(String.t())
  def load_texts(corpus_directory, opts \\ []) do
    max_files = Keyword.get(opts, :max_files)
    min_length = Keyword.get(opts, :min_length, 10)
    max_length = Keyword.get(opts, :max_length)
    shuffle = Keyword.get(opts, :shuffle, false)

    files = list_files(corpus_directory)

    files =
      if shuffle do
        Enum.shuffle(files)
      else
        files
      end

    files =
      if max_files do
        Enum.take(files, max_files)
      else
        files
      end

    files
    |> Enum.map(&File.read!/1)
    |> Enum.filter(fn text ->
      length = String.length(text)
      length >= min_length && (is_nil(max_length) || length <= max_length)
    end)
  end

  @doc """
  Loads texts and splits them into chunks of approximately equal size.

  This is useful for creating training examples from large corpus files.

  ## Parameters

  - `corpus_directory`: Path to the corpus directory
  - `opts`: Options
    - `:chunk_size` - Target size for each chunk in characters (default: 500)
    - `:overlap` - Number of overlapping characters between chunks (default: 50)
    - All options from `load_texts/2`

  ## Examples

      iex> chunks = MachineLearning.Corpus.load_chunked_texts(
      ...>   "./tmp/corpus",
      ...>   chunk_size: 500,
      ...>   overlap: 50,
      ...>   max_files: 50
      ...> )
      iex> is_list(chunks)
      true
  """
  @spec load_chunked_texts(Path.t(), keyword()) :: list(String.t())
  def load_chunked_texts(corpus_directory, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, 500)
    overlap = Keyword.get(opts, :overlap, 50)

    corpus_directory
    |> load_texts(opts)
    |> Enum.flat_map(fn text ->
      chunk_text(text, chunk_size, overlap)
    end)
  end

  @doc """
  Loads texts and splits them by lines or paragraphs.

  ## Parameters

  - `corpus_directory`: Path to the corpus directory
  - `opts`: Options
    - `:split_by` - Split by :lines or :paragraphs (default: :lines)
    - All options from `load_texts/2`

  ## Examples

      iex> lines = MachineLearning.Corpus.load_split_texts(
      ...>   "./tmp/corpus",
      ...>   split_by: :lines,
      ...>   min_length: 20
      ...> )
      iex> is_list(lines)
      true
  """
  @spec load_split_texts(Path.t(), keyword()) :: list(String.t())
  def load_split_texts(corpus_directory, opts \\ []) do
    split_by = Keyword.get(opts, :split_by, :lines)
    min_length = Keyword.get(opts, :min_length, 10)

    corpus_directory
    |> load_texts(Keyword.delete(opts, :min_length))
    |> Enum.flat_map(fn text ->
      case split_by do
        :lines ->
          String.split(text, "\n")

        :paragraphs ->
          String.split(text, ~r/\n\s*\n/)
      end
    end)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.filter(&(String.length(&1) >= min_length))
  end

  @doc """
  Loads a sample of texts for quick experimentation.

  ## Parameters

  - `corpus_directory`: Path to the corpus directory
  - `count`: Number of samples to return (default: 100)

  ## Examples

      iex> samples = MachineLearning.Corpus.load_sample("./tmp/corpus", 50)
      iex> is_list(samples)
      true
  """
  @spec load_sample(Path.t(), integer()) :: list(String.t())
  def load_sample(corpus_directory, count \\ 100) do
    load_split_texts(corpus_directory,
      split_by: :lines,
      min_length: 30,
      shuffle: true
    )
    |> Enum.take(count)
  end

  @doc """
  Evaluates a corpus directory for training suitability with the given model.

  Performs both basic corpus analysis and detailed token frequency analysis
  using the model's tokenizer.

  ## Parameters

  - `corpus_directory`: Path to the corpus directory
  - `model`: A MachineLearning.Transformer.Model struct with a tokenizer

  ## Returns

  A map containing:
  - `:corpus_stats` - Basic statistics (files, characters, etc.)
  - `:token_stats` - Token-level statistics
    - `:total_tokens` - Total tokens after tokenization
    - `:unique_tokens` - Number of unique tokens found
    - `:token_frequencies` - Map of token_id => frequency
    - `:most_common_tokens` - Top 50 most frequent tokens
    - `:unused_tokens` - Tokens in vocabulary that never appear
    - `:coverage_ratio` - Ratio of vocab tokens that appear in corpus
  - `:model_config` - Model configuration extracted from the model
  - `:recommendations` - List of actionable recommendations

  ## Examples

      iex> model = MachineLearning.Transformer.Model.load("models/my_model")
      iex> eval = MachineLearning.Transformer.Corpus.evaluate("corpus/training", model)
      iex> eval.token_stats.total_tokens
      1_234_567
  """
  @spec evaluate(Path.t(), Model.t()) :: map()
  def evaluate(corpus_directory, %Model{tokenizer: tokenizer, config: config} = _model) do
    Logger.info("Starting corpus evaluation for: #{corpus_directory}")

    # Step 1: Basic corpus statistics
    corpus_stats = analyze_corpus_basic(corpus_directory)

    # Step 2: Tokenize corpus and analyze token frequencies
    token_stats = analyze_token_frequencies(corpus_directory, tokenizer)

    # Step 3: Extract model configuration
    model_config = extract_model_config(config, tokenizer)

    # Step 4: Generate recommendations
    recommendations = generate_evaluation_recommendations(corpus_stats, token_stats, model_config)

    %{
      corpus_stats: corpus_stats,
      token_stats: token_stats,
      model_config: model_config,
      recommendations: recommendations,
      adequacy: determine_adequacy(token_stats, model_config),
      suitability_score: calculate_score(corpus_stats, token_stats, model_config)
    }
  end

  # Analyzes basic corpus statistics (files, characters, etc.)
  defp analyze_corpus_basic(corpus_directory) do
    files = list_files(corpus_directory) |> Enum.to_list()

    file_stats =
      files
      |> Task.async_stream(
        fn filename ->
          content = File.read!(filename)

          %{
            char_count: String.length(content),
            content_hash: :crypto.hash(:md5, content) |> Base.encode16()
          }
        end,
        max_concurrency: @max_concurrency,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, stat} -> stat end)

    total_files = length(file_stats)
    total_chars = Enum.sum(Enum.map(file_stats, & &1.char_count))
    avg_file_size = if total_files > 0, do: div(total_chars, total_files), else: 0

    unique_hashes = file_stats |> Enum.map(& &1.content_hash) |> Enum.uniq() |> length()
    diversity_ratio = if total_files > 0, do: unique_hashes / total_files, else: 0.0

    %{
      total_files: total_files,
      total_chars: total_chars,
      avg_file_size: avg_file_size,
      diversity_ratio: diversity_ratio
    }
  end

  # Tokenizes corpus and analyzes token frequencies
  defp analyze_token_frequencies(corpus_directory, tokenizer) do
    Logger.info("Tokenizing corpus files for frequency analysis...")

    files = list_files(corpus_directory) |> Enum.to_list()
    vocab_size = Tokenizer.vocab_size(tokenizer)

    # Tokenize all files and collect token frequencies
    token_frequencies =
      files
      |> Task.async_stream(
        fn filename ->
          content = File.read!(filename)
          token_ids = Tokenizer.encode(tokenizer, content)
          Enum.frequencies(token_ids)
        end,
        max_concurrency: @max_concurrency,
        timeout: :infinity
      )
      |> Enum.reduce(%{}, fn {:ok, freq_map}, acc ->
        Map.merge(acc, freq_map, fn _k, v1, v2 -> v1 + v2 end)
      end)

    total_tokens = Enum.sum(Map.values(token_frequencies))
    unique_tokens = map_size(token_frequencies)

    # Find most common tokens
    most_common =
      token_frequencies
      |> Enum.sort_by(fn {_token_id, freq} -> -freq end)
      |> Enum.take(50)
      |> Enum.map(fn {token_id, freq} ->
        token_text = Map.get(tokenizer.id_to_token, token_id, "<UNKNOWN>")
        %{token_id: token_id, token: token_text, frequency: freq}
      end)

    # Find unused tokens (in vocabulary but never appear in corpus)
    all_token_ids = MapSet.new(0..(vocab_size - 1))
    used_token_ids = MapSet.new(Map.keys(token_frequencies))
    unused_token_ids = MapSet.difference(all_token_ids, used_token_ids) |> MapSet.to_list()

    coverage_ratio = unique_tokens / vocab_size
    avg_frequency = if unique_tokens > 0, do: total_tokens / unique_tokens, else: 0

    %{
      total_tokens: total_tokens,
      unique_tokens: unique_tokens,
      token_frequencies: token_frequencies,
      most_common_tokens: most_common,
      unused_tokens: length(unused_token_ids),
      unused_token_ids: Enum.take(unused_token_ids, 100),
      coverage_ratio: coverage_ratio,
      avg_token_frequency: Float.round(avg_frequency, 2),
      tokens_per_vocab_item: Float.round(total_tokens / vocab_size, 2)
    }
  end

  # Extracts model configuration from config map
  defp extract_model_config(config, tokenizer) do
    %{
      vocab_size: Tokenizer.vocab_size(tokenizer),
      max_seq_len: Map.get(config, "max_seq_len", Map.get(config, :max_seq_len, 256)),
      embed_dim: Map.get(config, "embed_dim", Map.get(config, :embed_dim, 256)),
      num_layers: Map.get(config, "num_layers", Map.get(config, :num_layers, 4)),
      num_heads: Map.get(config, "num_heads", Map.get(config, :num_heads, 8))
    }
  end

  # Generates recommendations based on analysis
  defp generate_evaluation_recommendations(corpus_stats, token_stats, model_config) do
    recommendations = []

    # Check total token count
    min_tokens = model_config.vocab_size * 100
    optimal_tokens = model_config.vocab_size * 250

    recommendations =
      cond do
        token_stats.total_tokens < model_config.vocab_size * 25 ->
          [
            "⚠️  CRITICAL: Corpus has only #{format_number(token_stats.total_tokens)} tokens. Need at least #{format_number(min_tokens)} for training."
            | recommendations
          ]

        token_stats.total_tokens < min_tokens ->
          [
            "⚠️  WARNING: Corpus is small (#{format_number(token_stats.total_tokens)} tokens). Recommend #{format_number(optimal_tokens)} for optimal results."
            | recommendations
          ]

        token_stats.total_tokens < optimal_tokens ->
          [
            "ℹ️  Corpus is adequate but larger would be better (current: #{format_number(token_stats.total_tokens)}, optimal: #{format_number(optimal_tokens)} tokens)."
            | recommendations
          ]

        true ->
          recommendations
      end

    # Check vocabulary coverage
    recommendations =
      cond do
        token_stats.coverage_ratio < 0.5 ->
          [
            "⚠️  CRITICAL: Only #{Float.round(token_stats.coverage_ratio * 100, 1)}% of vocabulary appears in corpus. Many tokens are unused!"
            | recommendations
          ]

        token_stats.coverage_ratio < 0.7 ->
          [
            "⚠️  WARNING: Only #{Float.round(token_stats.coverage_ratio * 100, 1)}% of vocabulary is used. Consider more diverse content."
            | recommendations
          ]

        token_stats.coverage_ratio < 0.9 ->
          [
            "ℹ️  Vocabulary coverage is #{Float.round(token_stats.coverage_ratio * 100, 1)}%. Good, but could be more comprehensive."
            | recommendations
          ]

        true ->
          recommendations
      end

    # Check content diversity
    recommendations =
      if corpus_stats.diversity_ratio < 0.7 do
        [
          "⚠️  Low content diversity (#{Float.round(corpus_stats.diversity_ratio * 100, 1)}%). Many duplicate files detected."
          | recommendations
        ]
      else
        recommendations
      end

    # Check average token frequency
    recommendations =
      if token_stats.avg_token_frequency < 10 do
        [
          "⚠️  Low average token frequency (#{token_stats.avg_token_frequency}). Tokens appear too rarely for effective learning."
          | recommendations
        ]
      else
        recommendations
      end

    Enum.reverse(recommendations)
  end

  # Determines adequacy level
  defp determine_adequacy(token_stats, model_config) do
    tokens_per_vocab = token_stats.total_tokens / model_config.vocab_size

    cond do
      tokens_per_vocab < 25 -> :insufficient
      tokens_per_vocab < 100 -> :minimal
      tokens_per_vocab < 250 -> :adequate
      true -> :excellent
    end
  end

  # Calculates overall suitability score
  defp calculate_score(corpus_stats, token_stats, model_config) do
    # Token coverage score (0-40)
    tokens_per_vocab = token_stats.total_tokens / model_config.vocab_size

    coverage_score =
      cond do
        tokens_per_vocab >= 250 -> 40
        tokens_per_vocab >= 100 -> 30 + (tokens_per_vocab - 100) / 150 * 10
        tokens_per_vocab >= 25 -> 10 + (tokens_per_vocab - 25) / 75 * 20
        true -> tokens_per_vocab / 25 * 10
      end

    # Vocabulary usage score (0-30)
    vocab_score = token_stats.coverage_ratio * 30

    # Diversity score (0-20)
    diversity_score = corpus_stats.diversity_ratio * 20

    # Frequency distribution score (0-10)
    freq_score =
      if token_stats.avg_token_frequency >= 25 do
        10
      else
        token_stats.avg_token_frequency / 25 * 10
      end

    total = coverage_score + vocab_score + diversity_score + freq_score
    Float.round(min(100, total), 1)
  end

  # Private functions

  defp chunk_text(text, chunk_size, overlap) do
    text_length = String.length(text)

    if text_length <= chunk_size do
      [text]
    else
      graphemes = String.graphemes(text)
      step = chunk_size - overlap

      0..text_length
      |> Stream.take_every(step)
      |> Stream.map(fn start ->
        graphemes
        |> Enum.slice(start, chunk_size)
        |> Enum.join()
      end)
      |> Enum.reject(&(&1 == ""))
    end
  end

  @spec list_files(Path.t()) :: Enumerable.t()
  defp list_files(directory) do
    File.ls!(directory)
    |> Enum.reject(&String.starts_with?(&1, "."))
    |> Enum.reject(&String.ends_with?(&1, "_modules"))
    |> Enum.reject(&String.ends_with?(&1, "bench"))
    |> Stream.flat_map(fn name ->
      full_name = Path.join(directory, name)

      case File.dir?(full_name) do
        true -> list_files(full_name)
        false -> [full_name]
      end
    end)
  end

  @spec corpus_file_path(String.t()) :: Path.t()
  defp corpus_file_path(<<dir::bytes-2, filename::binary>>) do
    Path.join(dir, filename)
  end

  defp filename_header(filename) do
    """
    Filename: #{filename}
    ---------------------------------------------
    """
  end

  defp format_number(num) when num >= 1_000_000 do
    "#{Float.round(num / 1_000_000, 1)}M"
  end

  defp format_number(num) when num >= 1_000 do
    "#{Float.round(num / 1_000, 1)}K"
  end

  defp format_number(num), do: "#{num}"
end
