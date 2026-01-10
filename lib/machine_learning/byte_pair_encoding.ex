defmodule MachineLearning.BytePairEncoding do
  require Logger
  alias MachineLearning.BytePairEncoding.Token

  @moduledoc """
  Module for Byte Pair Encoding (BPE) tokenization.

  This module provides functionalities to create a corpus from source files,
  compress the corpus using BPE, and encode new data using the generated tokens.


  ## Example

      iex> source_dir = "path/to/source/files"
      iex> corpus_dir = "path/to/corpus"
      iex> MachineLearning.BytePairEncoding.add_to_corpus(source_dir, corpus_dir)
      :ok
      iex> expected_vocab_size = 1000
      iex> tokens = MachineLearning.BytePairEncoding.compress(corpus_dir, expected_vocab_size)


  You can recover from previous run by loading saved tokens:

      iex> saved_tokens = MachineLearning.BytePairEncoding.load("path/to/saved_tokens.bert")
      iex> tokens = MachineLearning.BytePairEncoding.compress(corpus_dir, expected_vocab_size, %{initial_tokens: saved_tokens})


  You can also specify where to save the cache during compression:

      iex> tokens = MachineLearning.BytePairEncoding.compress(corpus_dir, expected_vocab_size, %{save_path: "path/to/saved_tokens.bert"})
  """

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

  @max_concurrency 10

  def save(tokens, filename) do
    File.write!(filename, :erlang.term_to_binary(tokens))
  end

  def load(filename) do
    File.read!(filename)
    |> :erlang.binary_to_term()
  end

  @spec add_to_corpus(Path.t(), Path.t()) :: :ok
  def add_to_corpus(source_directory, corpus_directory, opts \\ %{}) do
    accepted_extensions = Map.get(opts, :extensions, @default_extensions)
    File.mkdir_p!(corpus_directory)

    list_files(source_directory)
    |> Stream.filter(fn filename -> Path.extname(filename) in accepted_extensions end)
    |> Task.async_stream(fn filename ->
      content = filename_header(filename) <> File.read!(filename)
      corpus_name = :crypto.hash(:sha256, content) |> Base.encode16() |> String.downcase()
      corpus_file_path = Path.join(corpus_directory, corpus_file_path(corpus_name))
      File.mkdir_p!(Path.dirname(corpus_file_path))
      File.write!(corpus_file_path, content)
    end)
    |> Stream.run()
  end

  @spec list_files(Path.t()) :: list(Path.t())
  defp list_files(directory) do
    File.ls!(directory)
    |> Enum.reject(&String.starts_with?(&1, "."))
    # Reject node_modules and similar directories
    |> Enum.reject(&String.ends_with?(&1, "_modules"))
    |> Stream.flat_map(fn name ->
      full_name = Path.join(directory, name)

      case File.dir?(full_name) do
        true -> list_files(full_name)
        false -> [full_name]
      end
    end)
  end

  def compress(corpus_directory, expected_vocabulary_size, opts \\ %{}) do
    tokens = Map.get(opts, :initial_tokens, [])
    save_path = Map.get(opts, :save_path, nil)
    cache_file = Map.get(opts, :cache_file, nil)

    setup_or_load_cache(corpus_directory, cache_file, tokens)
    |> do_compress(expected_vocabulary_size, save_path)
  end

  defp setup_or_load_cache(corpus_directory, cache_file, tokens) do
    cond do
      is_nil(cache_file) or not File.exists?(cache_file) ->
        Logger.info("Setting up new cache...")

        :ets.new(:byte_pair_encoding_cache, [:set, :public])
        |> tap(fn _ -> Logger.info("Created new cache for compression...") end)
        |> insert_filenames_to_cache(corpus_directory)
        |> tap(fn _ -> Logger.info("Inserted filenames to cache...") end)
        |> insert_tokens_to_cache(tokens)
        |> tap(fn _ -> Logger.info("Inserted initial tokens to cache...") end)
        |> populate_cache_with_tokenized_files(tokens)
        |> tap(fn _ -> Logger.info("Populated cache with tokenized files...") end)

      true ->
        Logger.info("Cache loaded from file, skipping setup...")
        :ets.file2tab(String.to_charlist(cache_file))
    end
  end

  defp do_compress(cache, expected_vocabulary_size, save_path) do
    tokens = get_tokens_from_cache(cache)

    {frequencies, elements} =
      cache
      |> calculate_frequency()

    vocabulary_size = MapSet.size(elements)
    highest_freq = frequencies |> Enum.at(0) |> elem(1) |> Map.get(:freq)
    highest_freq_tokens = frequencies |> Enum.at(0) |> elem(1) |> Map.get(:tokens)
    highest_freq_token = frequencies |> Enum.at(0) |> elem(0)
    stop? = vocabulary_size >= expected_vocabulary_size or highest_freq < 2

    Logger.info(
      "Current vocabulary size: #{vocabulary_size}, highest frequency: #{highest_freq}."
    )

    if stop? do
      iterate_cache(cache)
      |> Enum.flat_map(fn content ->
        Enum.uniq(content)
      end)
      |> Enum.uniq()
      |> Enum.map(fn
        %Token{} = token -> token
        byte -> Token.new(byte)
      end)
      |> tap(fn _ ->
        :ets.delete(cache)
      end)
    else
      new_tokens =
        frequencies
        |> tl()
        |> Enum.take_while(fn {_token, %{freq: freq, tokens: tokens}} ->
          freq > highest_freq / 2 and
            MapSet.disjoint?(MapSet.new(tokens), MapSet.new(highest_freq_tokens))
        end)
        |> Enum.map(fn {token, _freq} -> token end)

      new_tokens = [highest_freq_token | new_tokens]

      Logger.info("New tokens to add: #{inspect(new_tokens)}")

      retokenize_cache(cache, new_tokens)
      insert_tokens_to_cache(cache, new_tokens ++ tokens)

      if save_path do
        persist_table(cache, save_path)
        Logger.info("Cache saved to #{save_path}")
      end

      do_compress(cache, expected_vocabulary_size, save_path)
    end
  end

  # TODO optimize
  defp generate_frequency(content, _) do
    content
    # Pair frequency counting
    |> case do
      [_ | sliced] = graphemes ->
        Enum.zip(graphemes, sliced)
        |> then(fn pairs ->
          pairs
        end)
        |> Enum.frequencies()

      _ ->
        %{}
    end
  end

  defp insert_filenames_to_cache(cache, corpus_directory) do
    filenames = list_files(corpus_directory) |> Enum.to_list()
    :ets.insert(cache, {:__filenames__, filenames})
    cache
  end

  defp insert_tokens_to_cache(cache, tokens) do
    :ets.insert(cache, {:__tokens__, tokens})
    cache
  end

  defp populate_cache_with_tokenized_files(cache, tokens) do
    start = System.monotonic_time(:millisecond)
    file_count = list_files_from_cache(cache) |> Enum.count()
    tokenizer = MachineLearning.Tokenizer.from_vocab(tokens)

    IO.write("Populated 0/#{file_count}")

    list_files_from_cache(cache)
    |> Stream.chunk_every(250)
    |> Stream.with_index()
    |> Task.async_stream(
      fn {chunk, index} ->
        Enum.each(chunk, fn filename ->
          content = File.read!(filename) |> String.graphemes() |> tokenize(tokenizer)
          :ets.insert(cache, {filename, content})
        end)

        IO.write("\rPopulated #{index * 250}/#{file_count}")
      end,
      max_concurrency: @max_concurrency,
      timeout: :infinity
    )
    |> Stream.run()

    Logger.info(
      "Cache populated with tokenized files in #{System.monotonic_time(:millisecond) - start} ms."
    )

    cache
  end

  defp calculate_frequency(cache) do
    start = System.monotonic_time(:millisecond)

    iterate_cache(cache)
    |> Stream.chunk_every(250)
    |> Task.async_stream(
      fn chunk ->
        Enum.reduce(chunk, {%{}, MapSet.new()}, fn content, {freq_acc, elements_acc} ->
          freq_map = generate_frequency(content, nil)
          elements = content |> MapSet.new()
          freq_acc = Map.merge(freq_acc, freq_map, fn _key, val1, val2 -> val1 + val2 end)
          {freq_acc, MapSet.union(elements_acc, elements)}
        end)
      end,
      max_concurrency: @max_concurrency,
      timeout: :infinity
    )
    |> Enum.reduce({%{}, MapSet.new()}, fn {:ok, {freq_map, elements}},
                                           {acc_freq, acc_elements} ->
      {
        Map.merge(acc_freq, freq_map, fn _key, val1, val2 -> val1 + val2 end),
        MapSet.union(acc_elements, elements)
      }
    end)
    |> then(fn {freq_map, elements} ->
      {Map.new(freq_map, fn {{a, b}, freq} ->
         {Token.new(a, b), %{freq: freq, tokens: [a, b]}}
       end)
       |> Enum.sort_by(fn {_token, %{freq: freq}} -> -freq end), elements}
    end)
    |> tap(fn _ ->
      Logger.info("Frequencies calculated in #{System.monotonic_time(:millisecond) - start} ms.")
    end)
  end

  defp retokenize_cache(cache, new_tokens) do
    start = System.monotonic_time(:millisecond)

    list_files_from_cache(cache)
    |> Stream.chunk_every(250)
    |> Task.async_stream(
      fn chunk ->
        Enum.each(chunk, fn filename ->
          [{_, content}] =
            :ets.lookup(cache, filename)

          new_content = simple_tokenize(content, new_tokens)
          :ets.insert(cache, {filename, new_content})
        end)
      end,
      max_concurrency: @max_concurrency,
      timeout: :infinity
    )
    |> Stream.run()
    |> tap(fn _ ->
      Logger.info("Cache retokenized in #{System.monotonic_time(:millisecond) - start} ms.")
      cache
    end)
  end

  defp simple_tokenize(content, tokens) do
    Enum.reduce(content, {nil, []}, fn b, {prev, acc} ->
      cond do
        is_nil(prev) ->
          {b, acc}

        token = Enum.find(tokens, fn token -> "#{prev}#{b}" == token.value end) ->
          {nil, [token.value | acc]}

        true ->
          {b, [prev | acc]}
      end
    end)
    |> then(fn {prev, acc} -> [prev | acc] end)
    |> Enum.reject(&is_nil/1)
    |> Enum.reverse()
  end

  defp tokenize(content, tokenizer) do
    MachineLearning.Tokenizer.tokenize(content, tokenizer)
  end

  defp iterate_cache(cache) do
    list_files_from_cache(cache)
    |> Stream.map(&get_from_cache(cache, &1))
  end

  defp list_files_from_cache(cache) do
    [{:__filenames__, filenames}] = :ets.lookup(cache, :__filenames__)
    filenames
  end

  defp get_from_cache(cache, filename) do
    [{_, content}] = :ets.lookup(cache, filename)
    content
  end

  defp get_tokens_from_cache(cache) do
    case :ets.lookup(cache, :__tokens__) do
      [{_, tokens}] -> tokens
      [] -> []
    end
  end

  defp persist_table(cache, filename) do
    :ets.tab2file(cache, String.to_charlist(filename))
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
end
