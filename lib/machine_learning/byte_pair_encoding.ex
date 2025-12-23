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
    ".py"
  ]

  @max_concurrency 50

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
    |> Stream.flat_map(fn name ->
      full_name = Path.join(directory, name)

      case File.dir?(full_name) do
        true -> list_files(full_name)
        false -> [full_name]
      end
    end)
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

    Logger.info("Pair #{inspect(token.value)} added to vocabulary.")

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
    itereate_cache(cache)
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
  defp generate_frequency(content, token) do
    content
    # Pair frequency counting
    |> case do
      [_ | sliced] = graphemes ->
        Enum.zip(graphemes, sliced)
        |> then(fn pairs ->
          if token do
            Enum.filter(pairs, fn {a, b} -> a == token.value or b == token.value end)
          else
            pairs
          end
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

  defp itereate_cache(cache) do
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
