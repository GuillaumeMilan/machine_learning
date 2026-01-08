defmodule MachineLearning.BytePairEncoding do
  require Logger
  alias MachineLearning.BytePairEncoding.Token

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

  @max_concurrency 5

  def save(tokens, filename) do
    File.write!(filename, :erlang.term_to_binary(tokens))
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

  def compress(corpus_directory, expected_vocabulary_size, tokens \\ []) do
    create_cache()
    |> tap(fn _ -> Logger.info("Created new cache for compression...") end)
    |> insert_filenames_to_cache(corpus_directory)
    |> tap(fn _ -> Logger.info("Inserted filenames to cache...") end)
    |> populate_cache_with_tokenized_files(tokens)
    |> tap(fn _ -> Logger.info("Populated cache with tokenized files...") end)
    |> do_compress(expected_vocabulary_size, tokens)
  end

  defp do_compress(cache, expected_vocabulary_size, tokens) do
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
      File.write!("/tmp/bpe_save.bert", :erlang.term_to_binary(new_tokens ++ tokens))
      Logger.info("Tokens saved to /tmp/bpe_save.bert")

      retokenize_cache(cache, new_tokens)
      do_compress(cache, expected_vocabulary_size, new_tokens ++ tokens)
    end
  end

  def encode(corpus_directory, vocab_size, bytes_map \\ []) do
    Logger.info("Generating Byte Pair Encoding with vocab size #{vocab_size}...")
    cache = generate_cache(corpus_directory, bytes_map)
    Logger.info("Cache populated...")

    do_encode(cache, vocab_size, bytes_map)
  end

  defp do_encode(cache, vocab_size, bytes_map) do
    Logger.info("Encoding step with current vocab size #{length(bytes_map)}...")

    token =
      get_frequencies_from_cache(cache)
      |> Enum.max_by(fn {_token, freq} -> freq end)
      |> then(fn {token, _freq} -> token end)

    bytes_map = [token | bytes_map]

    Logger.info("Selected token: #{inspect(token.value)}")

    if length(bytes_map) < vocab_size do
      refresh_cache(cache, [token])
      refresh_frequencies(cache, token)
      do_encode(cache, vocab_size, bytes_map)
    else
      :ets.delete(cache)
      bytes_map
    end
  end

  defp generate_corpus_frequencies(cache, token \\ nil) do
    iterate_cache(cache)
    |> Stream.chunk_every(250)
    |> Task.async_stream(
      fn chunk ->
        Enum.reduce(chunk, %{}, fn content, acc ->
          freq_map = generate_frequency(content, token)
          Map.merge(acc, freq_map, fn _key, val1, val2 -> val1 + val2 end)
        end)
      end,
      max_concurrency: @max_concurrency,
      timeout: :infinity
    )
    |> Enum.reduce(%{}, fn {:ok, freq_map}, acc ->
      Map.merge(acc, freq_map, fn _key, val1, val2 -> val1 + val2 end)
    end)
    |> Map.new(fn {{a, b}, freq} ->
      {Token.new(a, b), freq}
    end)
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

  defp generate_cache(corpus_directory, bytes_map) do
    start = System.monotonic_time(:millisecond)
    Logger.info("Generating cache from corpus directory #{corpus_directory}...")
    cache = :ets.new(:byte_pair_encoding_cache, [:set, :public])

    filenames = list_files(corpus_directory) |> Enum.to_list()
    :ets.insert(cache, {:__filenames__, filenames})

    filenames
    |> Stream.chunk_every(250)
    |> Task.async_stream(
      fn chunk ->
        Enum.each(chunk, fn filename ->
          content = File.read!(filename) |> String.graphemes()
          :ets.insert(cache, {filename, content})
        end)
      end,
      max_concurrency: @max_concurrency,
      timeout: :infinity
    )
    |> Stream.run()

    Logger.info("Cache populated in #{System.monotonic_time(:millisecond) - start} ms.")

    bytes_map
    |> Enum.reverse()
    |> then(&refresh_cache(cache, &1))

    frequencies = generate_corpus_frequencies(cache)
    :ets.insert(cache, {:__frequencies__, frequencies})

    Logger.info("Cache generated in #{System.monotonic_time(:millisecond) - start} ms.")
    cache
  end

  defp create_cache() do
    :ets.new(:byte_pair_encoding_cache, [:set, :public])
  end

  defp insert_filenames_to_cache(cache, corpus_directory) do
    filenames = list_files(corpus_directory) |> Enum.to_list()
    :ets.insert(cache, {:__filenames__, filenames})
    cache
  end

  defp populate_cache_with_tokenized_files(cache, tokens) do
    start = System.monotonic_time(:millisecond)

    list_files_from_cache(cache)
    |> Stream.chunk_every(250)
    |> Task.async_stream(
      fn chunk ->
        Enum.each(chunk, fn filename ->
          content = File.read!(filename) |> String.graphemes() |> tokenize(tokens)
          :ets.insert(cache, {filename, content})
        end)
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
          {nil, [token | acc]}

        true ->
          {b, [prev | acc]}
      end
    end)
    |> then(fn {prev, acc} -> [prev | acc] end)
    |> Enum.reject(&is_nil/1)
    |> Enum.reverse()
  end

  defp refresh_cache(cache, tokens) do
    start = System.monotonic_time(:millisecond)

    list_files_from_cache(cache)
    |> Stream.chunk_every(250)
    |> Task.async_stream(
      fn chunk ->
        Enum.each(chunk, fn filename ->
          [{_, content}] =
            :ets.lookup(cache, filename)

          new_content =
            case tokens do
              [token] -> replace_token(content, token)
              _ -> tokenize(content, tokens)
            end

          :ets.insert(cache, {filename, new_content})
        end)
      end,
      max_concurrency: @max_concurrency,
      timeout: :infinity
    )
    |> Stream.run()
    |> tap(fn _ ->
      Logger.info("Cache content refreshed in #{System.monotonic_time(:millisecond) - start} ms.")
    end)
  end

  defp refresh_frequencies(cache, token) do
    start = System.monotonic_time(:millisecond)

    current_frequencies = Map.delete(get_frequencies_from_cache(cache), token)

    frequencies = generate_corpus_frequencies(cache, token)
    new_frequencies = Map.merge(frequencies, current_frequencies)
    :ets.insert(cache, {:__frequencies__, new_frequencies})
    Logger.info("Frequencies refreshed in #{System.monotonic_time(:millisecond) - start} ms.")
  end

  @spec tokenize(list(String.t()), list(Token.t())) :: list(String.t() | Token.t())
  def tokenize(content, original_tokens) do
    content
    |> Enum.reduce({[], original_tokens, []}, fn b, {prevs, tokens, acc} ->
      prevs_str = prevs |> Enum.reverse() |> Enum.join()
      combined = "#{prevs_str}#{b}"

      tokens
      |> Enum.reduce({nil, []}, fn token, {selected_token, acc} ->
        cond do
          combined == token.value ->
            {token, acc}

          String.starts_with?(token.value, combined) ->
            {selected_token, [token | acc]}

          true ->
            {selected_token, acc}
        end
      end)
      |> case do
        # Can be merged into a single token
        {%Token{} = token, tokens} ->
          {[token], tokens, acc}

        # No matching token possible in the future, flush prevs
        {nil, []} ->
          {[b], original_tokens, prevs ++ acc}

        # Still possible to merge with future bytes
        {nil, tokens} ->
          {[b | prevs], tokens, acc}
      end
    end)
    |> then(fn {prevs, _, acc} -> prevs ++ acc end)
    |> Enum.reverse()
  end

  def clean_tokens(tokens, corpus_directory) do
    cache = generate_cache(corpus_directory, tokens)

    existing_tokens =
      get_frequencies_from_cache(cache)
      |> Map.keys()
      |> MapSet.new()

    :ets.delete(cache)

    Enum.filter(tokens, fn token -> MapSet.member?(existing_tokens, token) end)
  end

  defp replace_token([a | rem], token) do
    Enum.reduce(rem, {a, []}, fn b, {prev, acc} ->
      if "#{prev}#{b}" == token.value do
        {token, acc}
      else
        {b, [prev | acc]}
      end
    end)
    |> then(fn {last, acc} -> [last | acc] end)
    |> Enum.reverse()
  end

  defp replace_token(rem, _pair), do: rem

  defp iterate_cache(cache) do
    list_files_from_cache(cache)
    |> Stream.map(&get_from_cache(cache, &1))
  end

  defp get_frequencies_from_cache(cache) do
    [{_, frequencies}] = :ets.lookup(cache, :__frequencies__)
    frequencies
  end

  defp list_files_from_cache(cache) do
    [{:__filenames__, filenames}] = :ets.lookup(cache, :__filenames__)
    filenames
  end

  defp get_from_cache(cache, filename) do
    [{_, content}] = :ets.lookup(cache, filename)
    content
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
