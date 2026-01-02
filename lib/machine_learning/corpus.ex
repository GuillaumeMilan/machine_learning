defmodule MachineLearning.Corpus do
  @moduledoc """
  Utilities for loading and preparing text data from corpus directories.

  This module helps load training texts from the corpus directory created
  by the BytePairEncoding module.
  """

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
  Prints statistics about the corpus.

  Returns `:ok` after printing statistics to stdout.
  """
  @spec stats(Path.t()) :: :ok
  def stats(corpus_directory) do
    texts = load_texts(corpus_directory)
    file_count = length(texts)

    char_counts = Enum.map(texts, &String.length/1)
    total_chars = Enum.sum(char_counts)
    avg_chars = if file_count > 0, do: div(total_chars, file_count), else: 0
    max_chars = Enum.max(char_counts, fn -> 0 end)
    min_chars = Enum.min(char_counts, fn -> 0 end)

    total_lines =
      texts
      |> Enum.map(fn text -> length(String.split(text, "\n")) end)
      |> Enum.sum()

    IO.puts("Corpus Statistics:")
    IO.puts("  Files: #{format_number(file_count)}")
    IO.puts("  Total characters: #{format_number(total_chars)}")
    IO.puts("  Total lines: #{format_number(total_lines)}")
    IO.puts("  Average file size: #{format_number(avg_chars)} chars")
    IO.puts("  Longest file: #{format_number(max_chars)} chars")
    IO.puts("  Shortest file: #{format_number(min_chars)} chars")
  end

  # Private functions

  defp list_files(directory) do
    case File.ls(directory) do
      {:ok, entries} ->
        entries
        |> Enum.reject(&String.starts_with?(&1, "."))
        |> Enum.flat_map(fn name ->
          full_path = Path.join(directory, name)

          if File.dir?(full_path) do
            list_files(full_path)
          else
            [full_path]
          end
        end)

      {:error, _} ->
        []
    end
  end

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

  defp format_number(num) do
    num
    |> to_string()
    |> String.graphemes()
    |> Enum.reverse()
    |> Enum.chunk_every(3)
    |> Enum.join(",")
    |> String.reverse()
  end
end
