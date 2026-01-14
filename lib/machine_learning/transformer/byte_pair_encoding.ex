defmodule MachineLearning.Transformer.BytePairEncoding do
  require Logger

  # 10MB
  @default_batch_size_bytes 10_000_000
  @default_num_workers 4
  @state_file "bpe_state.bert"
  @vocab_file "vocabulary.bert"
  @batches_dir "batches"
  @cache_table :bpe_batch_cache

  # Cache management functions
  @spec init_cache() :: :ok
  defp init_cache do
    case :ets.info(@cache_table) do
      :undefined ->
        :ets.new(@cache_table, [:named_table, :public, :set, {:read_concurrency, true}])
        Logger.info("💾 Initialized batch data cache")
        :ok

      # Table already exists
      _ ->
        :ok
    end
  end

  @spec clear_cache() :: :ok
  defp clear_cache do
    case :ets.info(@cache_table) do
      :undefined ->
        :ok

      _ ->
        :ets.delete(@cache_table)
        Logger.info("🗑️ Cleared batch data cache")
        :ok
    end
  end

  @spec get_batch_data(Path.t()) :: {:ok, term()} | {:error, term()}
  defp get_batch_data(batch_file) do
    case File.read(batch_file) do
      {:ok, content} ->
        try do
          batch_data = :erlang.binary_to_term(content)
          {:ok, batch_data}
        rescue
          e -> {:error, "Failed to decode batch: #{Exception.message(e)}"}
        end

      {:error, reason} ->
        {:error, "Failed to read batch file: #{reason}"}
    end

    # Try to get from cache first
    # case :ets.lookup(@cache_table, batch_file) do
    #   [{^batch_file, batch_data}] ->
    #     {:ok, batch_data}
    #   [] ->
    #     # Not in cache, read from disk and cache it
    #     case File.read(batch_file) do
    #       {:ok, content} ->
    #         try do
    #           batch_data = :erlang.binary_to_term(content)
    #           :ets.insert(@cache_table, {batch_file, batch_data})
    #           {:ok, batch_data}
    #         rescue
    #           e -> {:error, "Failed to decode batch: #{Exception.message(e)}"}
    #         end
    #       {:error, reason} -> {:error, "Failed to read batch file: #{reason}"}
    #     end
    # end
  end

  @spec put_batch_data(Path.t(), term()) :: :ok | {:error, term()}
  defp put_batch_data(batch_file, batch_data) do
    try do
      # Write to disk
      encoded = :erlang.term_to_binary(batch_data)
      File.write!(batch_file, encoded)

      # Update cache
      # :ets.insert(@cache_table, {batch_file, batch_data})
      :ok
    rescue
      e -> {:error, "Failed to save batch data: #{Exception.message(e)}"}
    end
  end

  @doc """
  Initialize BPE training state from a corpus directory.

  ## Parameters
  - state_directory: Directory where training state will be saved
  - corpus_directory: Directory containing text files for training
  - opts: Configuration options
    - :batch_size_bytes (default: 10MB) - Target size for file batches
    - :num_workers (default: 4) - Number of parallel workers for initialization

  ## Returns
  {:ok, state_path} | {:error, reason}
  """
  def init(state_directory, corpus_directory, opts \\ %{}) do
    init_start = System.monotonic_time(:millisecond)
    batch_size = Map.get(opts, :batch_size_bytes, @default_batch_size_bytes)
    num_workers = Map.get(opts, :num_workers, @default_num_workers)

    init_cache()

    Logger.info(
      "🚀 Starting BPE initialization - batch_size: #{batch_size} bytes, workers: #{num_workers}"
    )

    with :ok <- File.mkdir_p(state_directory),
         :ok <- File.mkdir_p(Path.join(state_directory, @batches_dir)),
         {:ok, corpus_files} <- collect_corpus_files(corpus_directory),
         {:ok, batches} <- create_file_batches(corpus_files, batch_size, num_workers),
         {:ok, tokenized_batches} <- tokenize_batches(batches, state_directory, num_workers),
         {:ok, initial_vocab} <- build_initial_vocabulary(tokenized_batches, num_workers),
         :ok <- save_initial_state(state_directory, initial_vocab, tokenized_batches) do
      init_end = System.monotonic_time(:millisecond)
      init_time = init_end - init_start

      Logger.info(
        "✅ BPE initialization completed in #{init_time}ms - vocabulary size: #{map_size(initial_vocab)}"
      )

      {:ok, Path.join(state_directory, @state_file)}
    else
      {:error, reason} -> {:error, reason}
      error -> {:error, "Initialization failed: #{inspect(error)}"}
    end
  end

  @doc """
  Run BPE training for a specified number of steps.

  ## Parameters
  - state_dir: Directory containing the BPE training state
  - steps: Number of BPE merge steps to perform
  - opts: Configuration options
    - :num_workers (default: 4) - Number of parallel workers

  ## Returns
  {:ok, %{
    steps_completed: integer(),
    vocabulary_path: string(),
    total_merges: integer(),
    final_vocab_size: integer(),
    processing_time_ms: integer()
  }} | {:error, reason}
  """
  def run(state_dir, steps, opts \\ %{}) do
    num_workers = Map.get(opts, :num_workers, @default_num_workers)
    start_time = System.monotonic_time(:millisecond)

    init_cache()
    Logger.info("🚀 Starting BPE training - #{steps} steps with #{num_workers} workers")

    with {:ok, state} <- load_state(state_dir),
         {:ok, final_state} <- run_bpe_steps(state, state_dir, steps, num_workers) do
      end_time = System.monotonic_time(:millisecond)
      processing_time = end_time - start_time

      vocab_path = Path.join(state_dir, @vocab_file)
      vocab_reduction = map_size(state.vocabulary) - map_size(final_state.vocabulary)

      Logger.info(
        "✅ BPE compression completed in #{processing_time}ms - vocabulary reduced by #{vocab_reduction} tokens (#{Float.round(processing_time / steps, 1)}ms/step)"
      )

      # Clear cache after training completion
      clear_cache()

      stats = %{
        steps_completed: steps,
        vocabulary_path: vocab_path,
        total_merges: length(final_state.merge_rules),
        final_vocab_size: map_size(final_state.vocabulary),
        processing_time_ms: processing_time
      }

      {:ok, stats}
    else
      error -> error
    end
  end

  # Private functions

  @spec collect_corpus_files(Path.t()) :: {:ok, list(Path.t())} | {:error, term()}
  defp collect_corpus_files(corpus_directory) do
    collect_start = System.monotonic_time(:millisecond)

    if File.exists?(corpus_directory) do
      files = list_files(corpus_directory)

      collect_end = System.monotonic_time(:millisecond)
      collect_time = collect_end - collect_start
      total_size = files |> Enum.map(&File.stat!(&1).size) |> Enum.sum()

      Logger.info(
        "📁 Collected #{length(files)} files (#{Float.round(total_size / (1024 * 1024), 1)}MB) in #{collect_time}ms"
      )

      {:ok, files}
    else
      {:error, "Corpus directory does not exist: #{corpus_directory}"}
    end
  end

  @spec list_files(Path.t()) :: list(Path.t())
  defp list_files(directory) do
    # More efficient directory traversal using File.stream!
    try do
      directory
      |> File.stream!([], :line)
      |> Stream.map(&String.trim/1)
      |> Stream.reject(&(&1 == ""))
      |> Stream.map(&Path.join(directory, &1))
      |> Stream.flat_map(fn path ->
        case File.dir?(path) do
          true -> list_files(path)
          false -> [path]
        end
      end)
      |> Enum.to_list()
    rescue
      _ ->
        # Fallback to File.ls! if File.stream! fails
        File.ls!(directory)
        |> Stream.map(&Path.join(directory, &1))
        |> Stream.flat_map(fn full_path ->
          case File.dir?(full_path) do
            true -> list_files(full_path)
            false -> [full_path]
          end
        end)
        |> Enum.to_list()
    end
  end

  @spec create_file_batches(list(Path.t()), integer(), integer()) ::
          {:ok, list(list({Path.t(), integer()}))} | {:error, term()}
  defp create_file_batches(files, target_batch_size, num_workers) do
    batch_start = System.monotonic_time(:millisecond)

    # Parallelize file size collection for better I/O performance
    file_sizes =
      files
      |> Task.async_stream(
        fn file ->
          case File.stat(file) do
            {:ok, %{size: size}} ->
              {file, size}

            {:error, reason} ->
              Logger.warning("Could not get size for file #{file}: #{reason}")
              {file, 0}
          end
        end,
        max_concurrency: num_workers,
        timeout: :infinity
      )
      |> Enum.map(fn {:ok, result} -> result end)

    batches = create_balanced_batches(file_sizes, target_batch_size)

    batch_end = System.monotonic_time(:millisecond)
    batch_time = batch_end - batch_start

    if length(batches) > 0 do
      avg_batch_size = file_sizes |> Enum.map(&elem(&1, 1)) |> Enum.sum() |> div(length(batches))

      Logger.info(
        "📦 Created #{length(batches)} balanced batches (avg #{Float.round(avg_batch_size / 1024, 1)}KB each) in #{batch_time}ms"
      )
    else
      Logger.info("📦 Created 0 batches from empty corpus in #{batch_time}ms")
    end

    {:ok, batches}
  end

  @spec create_balanced_batches(list({Path.t(), integer()}), integer()) ::
          list(list({Path.t(), integer()}))
  defp create_balanced_batches(file_sizes, target_batch_size) do
    file_sizes
    |> Stream.chunk_while(
      # {current_batch, current_size}
      {[], 0},
      fn {file, size}, {batch, current_size} ->
        new_size = current_size + size
        new_batch = [{file, size} | batch]

        if new_size >= target_batch_size do
          {:cont, Enum.reverse(new_batch), {[], 0}}
        else
          {:cont, {new_batch, new_size}}
        end
      end,
      fn
        {[], 0} -> {:cont, []}
        {batch, _size} -> {:cont, Enum.reverse(batch), []}
      end
    )
    |> Enum.to_list()
  end

  @spec tokenize_batches(list(list({Path.t(), integer()})), Path.t(), integer()) ::
          {:ok, list(Path.t())} | {:error, term()}
  defp tokenize_batches(batches, state_dir, num_workers) do
    tokenize_start = System.monotonic_time(:millisecond)
    batches_dir = Path.join(state_dir, @batches_dir)

    Logger.info(
      "🔤 Starting tokenization of #{length(batches)} batches with #{num_workers} workers..."
    )

    batch_paths =
      batches
      |> Enum.with_index()
      |> Task.async_stream(
        fn {batch, index} ->
          tokenize_and_save_batch(batch, batches_dir, index)
        end,
        max_concurrency: num_workers,
        timeout: :infinity
      )
      |> Enum.reduce_while({:ok, []}, fn
        {:ok, {:ok, batch_path}}, {:ok, acc} -> {:cont, {:ok, [batch_path | acc]}}
        {:ok, {:error, reason}}, _ -> {:halt, {:error, reason}}
        {:error, reason}, _ -> {:halt, {:error, reason}}
      end)

    case batch_paths do
      {:ok, paths} ->
        tokenize_end = System.monotonic_time(:millisecond)
        tokenize_time = tokenize_end - tokenize_start

        if length(batches) > 0 do
          Logger.info(
            "✅ Tokenization completed in #{tokenize_time}ms (#{Float.round(tokenize_time / length(batches), 1)}ms/batch)"
          )
        else
          Logger.info("✅ Tokenization completed in #{tokenize_time}ms (no batches to process)")
        end

        {:ok, Enum.reverse(paths)}

      error ->
        error
    end
  end

  @spec tokenize_and_save_batch(list({Path.t(), integer()}), Path.t(), integer()) ::
          {:ok, Path.t()} | {:error, term()}
  defp tokenize_and_save_batch(batch, batches_dir, index) do
    batch_file = Path.join(batches_dir, "batch_#{index}.bert")

    try do
      tokenized_content =
        batch
        |> Enum.flat_map(fn {file_path, _} ->
          case File.read(file_path) do
            {:ok, content} ->
              tokens = initial_tokenize(content)
              [%{file_path: file_path, tokens: tokens}]

            {:error, reason} ->
              Logger.warning("Failed to read file #{file_path}: #{reason}")
              []
          end
        end)

      case put_batch_data(batch_file, tokenized_content) do
        :ok -> {:ok, batch_file}
        {:error, reason} -> {:error, "Failed to process batch #{index}: #{reason}"}
      end
    rescue
      e -> {:error, "Failed to process batch #{index}: #{Exception.message(e)}"}
    end
  end

  @spec initial_tokenize(String.t()) :: list(String.t())
  defp initial_tokenize(text) do
    # Simple character-level tokenization as starting point
    text
    |> String.graphemes()
    |> Enum.filter(&(&1 != ""))
  end

  @spec build_initial_vocabulary(list(Path.t()), integer()) :: {:ok, map()} | {:error, term()}
  defp build_initial_vocabulary(batch_files, num_workers) do
    vocab_start = System.monotonic_time(:millisecond)

    Logger.info(
      "📚 Building initial vocabulary from #{length(batch_files)} batch files with #{num_workers} workers..."
    )

    result =
      batch_files
      |> Task.async_stream(
        fn batch_file -> build_vocabulary_from_batch(batch_file) end,
        max_concurrency: num_workers,
        timeout: :infinity
      )
      |> Enum.reduce_while({:ok, %{}}, fn
        {:ok, {:ok, batch_vocab}}, {:ok, acc_vocab} ->
          merged_vocab = Map.merge(acc_vocab, batch_vocab, fn _, v1, v2 -> v1 + v2 end)
          {:cont, {:ok, merged_vocab}}

        {:ok, {:error, reason}}, _ ->
          {:halt, {:error, reason}}

        {:error, reason}, _ ->
          {:halt, {:error, reason}}
      end)

    case result do
      {:ok, vocabulary} ->
        vocab_end = System.monotonic_time(:millisecond)
        vocab_time = vocab_end - vocab_start
        total_tokens = vocabulary |> Map.values() |> Enum.sum()

        Logger.info(
          "✅ Initial vocabulary built in #{vocab_time}ms - #{map_size(vocabulary)} unique tokens, #{total_tokens} total"
        )

        {:ok, vocabulary}

      error ->
        error
    end
  end

  @spec build_vocabulary_from_batch(Path.t()) :: {:ok, map()} | {:error, term()}
  defp build_vocabulary_from_batch(batch_file) do
    case get_batch_data(batch_file) do
      {:ok, batch_data} ->
        try do
          batch_vocab =
            batch_data
            |> Enum.reduce(%{}, fn file_entry, vocab ->
              tokens = file_entry.tokens

              tokens
              |> Enum.reduce(vocab, fn token, v ->
                Map.update(v, token, 1, &(&1 + 1))
              end)
            end)

          {:ok, batch_vocab}
        rescue
          e -> {:error, "Failed to process batch #{batch_file}: #{Exception.message(e)}"}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec save_initial_state(Path.t(), map(), list(Path.t())) :: :ok | {:error, term()}
  defp save_initial_state(state_dir, vocabulary, batch_files) do
    state = %{
      step: 0,
      vocabulary: vocabulary,
      merge_rules: [],
      batch_files: batch_files,
      created_at: DateTime.utc_now() |> DateTime.to_iso8601()
    }

    state_file = Path.join(state_dir, @state_file)
    vocab_file = Path.join(state_dir, @vocab_file)

    try do
      encoded_state = :erlang.term_to_binary(state)
      encoded_vocab = :erlang.term_to_binary(vocabulary)

      with :ok <- File.write(state_file, encoded_state),
           :ok <- File.write(vocab_file, encoded_vocab) do
        :ok
      else
        error -> {:error, "Failed to save initial state: #{inspect(error)}"}
      end
    rescue
      e -> {:error, "Failed to save initial state: #{Exception.message(e)}"}
    end
  end

  @spec load_state(Path.t()) :: {:ok, map()} | {:error, term()}
  defp load_state(state_dir) do
    state_file = Path.join(state_dir, @state_file)

    case File.read(state_file) do
      {:ok, content} ->
        try do
          state = :erlang.binary_to_term(content)
          {:ok, state}
        rescue
          e -> {:error, "Failed to decode state: #{Exception.message(e)}"}
        end

      {:error, reason} ->
        {:error, "Failed to read state file: #{reason}"}
    end
  end

  @spec run_bpe_steps(map(), Path.t(), integer(), integer()) :: {:ok, map()} | {:error, term()}
  defp run_bpe_steps(state, state_dir, steps, num_workers) do
    initial_step = state.step

    Enum.reduce_while(1..steps, state, fn step, current_state ->
      step_start = System.monotonic_time(:millisecond)
      current_step_num = current_state.step + 1

      Logger.info(
        "📈 BPE Step #{current_step_num}/#{initial_step + steps} (vocab size: #{map_size(current_state.vocabulary)})"
      )

      case perform_bpe_step(current_state, num_workers) do
        {:ok, new_state} ->
          updated_state = %{new_state | step: current_state.step + 1}

          # Save state after each step for recoverability
          case save_state(state_dir, updated_state) do
            :ok ->
              step_end = System.monotonic_time(:millisecond)
              total_step_time = step_end - step_start
              Logger.info("   ✅ Step #{current_step_num} total time: #{total_step_time}ms")
              {:cont, updated_state}

            {:error, reason} ->
              {:halt, {:error, "Failed to save state at step #{step}: #{reason}"}}
          end

        {:error, reason} ->
          {:halt, {:error, "Failed at step #{step}: #{reason}"}}
      end
    end)
    |> case do
      %{} = final_state -> {:ok, final_state}
      error -> error
    end
  end

  @spec perform_bpe_step(map(), integer()) :: {:ok, map()} | {:error, term()}
  defp perform_bpe_step(state, num_workers) do
    step_start = System.monotonic_time(:millisecond)

    with {:ok, pair_counts} <- count_pairs_parallel(state.batch_files, num_workers),
         {:ok, most_frequent_pair} <- find_most_frequent_pair(pair_counts),
         {:ok, new_token} <- create_merge_token(most_frequent_pair),
         {:ok, _updated_batches} <-
           apply_merge_to_batches(state.batch_files, most_frequent_pair, new_token, num_workers),
         {:ok, updated_vocab} <-
           update_vocabulary(state.vocabulary, new_token, pair_counts[most_frequent_pair]) do
      {token1, token2} = most_frequent_pair

      new_merge_rule = %{
        pair: [token1, token2],
        token: new_token,
        frequency: pair_counts[most_frequent_pair]
      }

      updated_state = %{
        state
        | vocabulary: updated_vocab,
          merge_rules: [new_merge_rule | state.merge_rules]
      }

      step_end = System.monotonic_time(:millisecond)
      step_time = step_end - step_start

      Logger.info(
        "   ⚡ Step completed in #{step_time}ms - merged #{inspect(token1)} + #{inspect(token2)} → #{inspect(new_token)} (#{pair_counts[most_frequent_pair]} occurrences)"
      )

      {:ok, updated_state}
    else
      error -> error
    end
  end

  @spec count_pairs_parallel(list(Path.t()), integer()) :: {:ok, map()} | {:error, term()}
  defp count_pairs_parallel(batch_files, num_workers) do
    count_start = System.monotonic_time(:millisecond)

    result =
      batch_files
      |> Task.async_stream(
        fn batch_file -> count_pairs_in_batch(batch_file) end,
        max_concurrency: num_workers,
        timeout: :infinity
      )
      |> Enum.reduce_while({:ok, %{}}, fn
        {:ok, {:ok, batch_pairs}}, {:ok, acc} ->
          merged_pairs = Map.merge(acc, batch_pairs, fn _, v1, v2 -> v1 + v2 end)
          {:cont, {:ok, merged_pairs}}

        {:ok, {:error, reason}}, _ ->
          {:halt, {:error, reason}}

        {:error, reason}, _ ->
          {:halt, {:error, reason}}
      end)

    case result do
      {:ok, pair_counts} ->
        count_end = System.monotonic_time(:millisecond)
        count_time = count_end - count_start

        Logger.info(
          "     📊 Pair counting completed in #{count_time}ms - found #{map_size(pair_counts)} unique pairs"
        )

        {:ok, pair_counts}

      error ->
        error
    end
  end

  @spec count_pairs_in_batch(Path.t()) :: {:ok, map()} | {:error, term()}
  defp count_pairs_in_batch(batch_file) do
    case get_batch_data(batch_file) do
      {:ok, batch_data} ->
        try do
          pair_counts =
            batch_data
            |> Enum.reduce(%{}, fn file_entry, acc ->
              tokens = file_entry.tokens
              count_pairs_in_tokens(tokens, acc)
            end)

          {:ok, pair_counts}
        rescue
          e -> {:error, "Failed to decode batch: #{Exception.message(e)}"}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec count_pairs_in_tokens(list(String.t()), map()) :: map()
  defp count_pairs_in_tokens(tokens, acc) do
    tokens
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.map(&List.to_tuple/1)
    |> Enum.frequencies()
    |> Map.merge(acc, fn _, v1, v2 -> v1 + v2 end)
  end

  @spec find_most_frequent_pair(map()) :: {:ok, {String.t(), String.t()}} | {:error, term()}
  defp find_most_frequent_pair(pair_counts) when map_size(pair_counts) > 0 do
    most_frequent = Enum.max_by(pair_counts, fn {_, count} -> count end)
    {:ok, elem(most_frequent, 0)}
  end

  defp find_most_frequent_pair(_), do: {:error, "No pairs found"}

  @spec create_merge_token({String.t(), String.t()}) :: {:ok, String.t()} | {:error, term()}
  defp create_merge_token({token1, token2}) do
    {:ok, token1 <> token2}
  end

  @spec apply_merge_to_batches(list(Path.t()), {String.t(), String.t()}, String.t(), integer()) ::
          {:ok, list(Path.t())} | {:error, term()}
  defp apply_merge_to_batches(batch_files, pair, new_token, num_workers) do
    merge_start = System.monotonic_time(:millisecond)

    Logger.info(
      "     🔄 Applying merge to #{length(batch_files)} batch files with #{num_workers} workers with new token #{inspect(new_token)}...)"
    )

    # Process batches in chunks to reduce memory pressure and improve cache locality
    chunk_size = max(div(length(batch_files), num_workers), 1)

    batch_files
    |> Enum.chunk_every(chunk_size)
    |> Task.async_stream(
      fn batch_chunk ->
        # Process chunk sequentially to maintain cache efficiency
        chunk_results =
          batch_chunk
          |> Enum.map(fn batch_file -> apply_merge_to_batch(batch_file, pair, new_token) end)

        # Check for errors in chunk
        Enum.reduce_while(chunk_results, {:ok, []}, fn
          {:ok, file}, {:ok, acc} -> {:cont, {:ok, [file | acc]}}
          {:error, reason}, _ -> {:halt, {:error, reason}}
        end)
      end,
      max_concurrency: num_workers,
      timeout: :infinity
    )
    |> Enum.reduce_while({:ok, []}, fn
      {:ok, {:ok, chunk_files}}, {:ok, acc} -> {:cont, {:ok, chunk_files ++ acc}}
      {:ok, {:error, reason}}, _ -> {:halt, {:error, reason}}
      {:error, reason}, _ -> {:halt, {:error, reason}}
    end)
    |> case do
      {:ok, files} ->
        merge_end = System.monotonic_time(:millisecond)
        merge_time = merge_end - merge_start

        Logger.info(
          "     🔄 Merge application completed in #{merge_time}ms (#{Float.round(merge_time / length(batch_files), 1)}ms/batch)"
        )

        {:ok, files}

      error ->
        error
    end
  end

  @spec apply_merge_to_batch(Path.t(), {String.t(), String.t()}, String.t()) ::
          {:ok, Path.t()} | {:error, term()}
  defp apply_merge_to_batch(batch_file, {token1, token2} = _pair, new_token) do
    case get_batch_data(batch_file) do
      {:ok, batch_data} ->
        try do
          # Stream processing to reduce memory pressure
          updated_data =
            batch_data
            |> Stream.map(fn file_entry ->
              tokens = file_entry.tokens
              # Early exit if tokens don't contain the pair
              # if Enum.any?(tokens, &(&1 == token1 or &1 == token2)) do
              updated_tokens = merge_tokens(tokens, token1, token2, new_token)
              %{file_path: file_entry.file_path, tokens: updated_tokens}
              # else
              #   # No change needed
              #   file_entry
              # end
            end)
            |> Enum.to_list()

          case put_batch_data(batch_file, updated_data) do
            :ok -> {:ok, batch_file}
            {:error, reason} -> {:error, reason}
          end
        rescue
          e -> {:error, "Failed to process batch: #{Exception.message(e)}"}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  @spec merge_tokens(list(String.t()), String.t(), String.t(), String.t()) :: list(String.t())
  defp merge_tokens(tokens, token1, token2, new_token) do
    # Ultra-fast merge using :lists.reverse to avoid Enum.reverse overhead
    {result, pending} =
      tokens
      |> Enum.reduce({[], nil}, fn current_token, {acc, pending_token} ->
        case {pending_token, current_token} do
          {^token1, ^token2} ->
            # Found the pair to merge, add merged token and clear pending
            {[new_token | acc], nil}

          {nil, token} ->
            # No pending token, current becomes pending
            {acc, token}

          {pending, token} ->
            # Add the pending token to result, current becomes new pending
            {[pending | acc], token}
        end
      end)

    # Add any remaining pending token and reverse using faster :lists.reverse
    final_list =
      case pending do
        nil -> result
        token -> [token | result]
      end

    :lists.reverse(final_list)
  end

  @spec update_vocabulary(map(), String.t(), integer()) :: {:ok, map()} | {:error, term()}
  defp update_vocabulary(vocabulary, new_token, frequency) do
    updated_vocab = Map.put(vocabulary, new_token, frequency)
    {:ok, updated_vocab}
  end

  @spec save_state(Path.t(), map()) :: :ok | {:error, term()}
  defp save_state(state_dir, state) do
    state_file = Path.join(state_dir, @state_file)
    vocab_file = Path.join(state_dir, @vocab_file)

    try do
      encoded_state = :erlang.term_to_binary(state)
      encoded_vocab = :erlang.term_to_binary(state.vocabulary)

      with :ok <- File.write(state_file, encoded_state),
           :ok <- File.write(vocab_file, encoded_vocab) do
        :ok
      else
        error -> {:error, "Failed to save state: #{inspect(error)}"}
      end
    rescue
      e -> {:error, "Failed to save state: #{Exception.message(e)}"}
    end
  end
end
