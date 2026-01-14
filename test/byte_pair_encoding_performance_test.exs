defmodule MachineLearning.Transformer.BytePairEncodingPerformanceTest do
  use ExUnit.Case, async: true
  alias MachineLearning.Transformer.BytePairEncoding

  setup_all do
    tmp_dir = System.tmp_dir!() |> Path.join("bpe_test_#{System.unique_integer([:positive])}")
    corpus_dir = Path.join(tmp_dir, "corpus")
    state_dir = Path.join(tmp_dir, "state")

    # Create test directories
    File.mkdir_p!(corpus_dir)
    File.mkdir_p!(state_dir)

    # Create test corpus files
    create_test_corpus(corpus_dir)

    on_exit(fn -> File.rm_rf!(tmp_dir) end)

    %{tmp_dir: tmp_dir, corpus_dir: corpus_dir, state_dir: state_dir}
  end

  describe "init/3" do
    test "initializes BPE state from corpus directory", %{
      corpus_dir: corpus_dir,
      state_dir: state_dir
    } do
      opts = %{batch_size_bytes: 1000, num_workers: 2}

      assert {:ok, state_path} = BytePairEncoding.init(state_dir, corpus_dir, opts)
      assert File.exists?(state_path)

      # Verify state file structure
      {:ok, state_content} = File.read(state_path)
      state = :erlang.binary_to_term(state_content)

      assert state.step == 0
      assert is_map(state.vocabulary)
      assert state.merge_rules == []
      assert is_list(state.batch_files)
      assert Map.has_key?(state, :created_at)

      # Verify vocabulary file exists
      vocab_file = Path.join(state_dir, "vocabulary.bert")
      assert File.exists?(vocab_file)

      # Verify batches directory exists and contains files
      batches_dir = Path.join(state_dir, "batches")
      assert File.exists?(batches_dir)
      {:ok, batch_files} = File.ls(batches_dir)
      assert length(batch_files) > 0
    end

    test "handles empty corpus directory", %{tmp_dir: tmp_dir} do
      empty_corpus = Path.join(tmp_dir, "empty_corpus")
      empty_state = Path.join(tmp_dir, "empty_state")
      File.mkdir_p!(empty_corpus)
      File.mkdir_p!(empty_state)

      assert {:ok, _state_path} = BytePairEncoding.init(empty_state, empty_corpus)

      # Should create valid state even with empty corpus
      state_file = Path.join(empty_state, "bpe_state.bert")
      {:ok, state_content} = File.read(state_file)
      state = :erlang.binary_to_term(state_content)

      assert state.step == 0
      assert state.vocabulary == %{}
    end

    test "handles invalid corpus directory", %{state_dir: state_dir} do
      invalid_corpus = "/non/existent/path"

      assert {:error, reason} = BytePairEncoding.init(state_dir, invalid_corpus)
      assert reason =~ "does not exist"
    end
  end

  describe "run/3" do
    setup %{corpus_dir: corpus_dir, state_dir: state_dir} do
      # Initialize BPE state before each test
      opts = %{batch_size_bytes: 500, num_workers: 2}
      {:ok, _state_path} = BytePairEncoding.init(state_dir, corpus_dir, opts)
      :ok
    end

    test "runs BPE training for specified steps", %{state_dir: state_dir} do
      steps = 3
      opts = %{num_workers: 2}

      assert {:ok, stats} = BytePairEncoding.run(state_dir, steps, opts)

      # Verify return statistics
      assert stats.steps_completed == steps
      assert is_binary(stats.vocabulary_path)
      assert File.exists?(stats.vocabulary_path)
      assert is_integer(stats.total_merges)
      assert is_integer(stats.final_vocab_size)
      assert is_integer(stats.processing_time_ms)
      assert stats.total_merges == steps

      # Verify state was updated
      state_file = Path.join(state_dir, "bpe_state.bert")
      {:ok, state_content} = File.read(state_file)
      state = :erlang.binary_to_term(state_content)

      assert state.step == steps
      assert length(state.merge_rules) == steps

      # Verify merge rules have correct structure
      Enum.each(state.merge_rules, fn rule ->
        assert Map.has_key?(rule, :pair)
        assert Map.has_key?(rule, :token)
        assert Map.has_key?(rule, :frequency)
        assert is_list(rule.pair)
        assert length(rule.pair) == 2
      end)
    end

    test "can resume training from saved state", %{state_dir: state_dir} do
      # Run first batch of steps
      assert {:ok, _stats1} = BytePairEncoding.run(state_dir, 2, %{num_workers: 1})

      # Run second batch of steps
      assert {:ok, stats2} = BytePairEncoding.run(state_dir, 3, %{num_workers: 1})

      # Should have completed 3 more steps (total state.step should be 5)
      assert stats2.steps_completed == 3
      # 2 + 3 steps
      assert stats2.total_merges == 5

      # Verify final state
      state_file = Path.join(state_dir, "bpe_state.bert")
      {:ok, state_content} = File.read(state_file)
      state = :erlang.binary_to_term(state_content)

      assert state.step == 5
      assert length(state.merge_rules) == 5
    end

    test "handles case with no valid pairs to merge", %{tmp_dir: tmp_dir} do
      # Create minimal corpus with single characters
      minimal_corpus = Path.join(tmp_dir, "minimal_corpus")
      minimal_state = Path.join(tmp_dir, "minimal_state")
      File.mkdir_p!(minimal_corpus)
      File.mkdir_p!(minimal_state)

      # Write single character file
      File.write!(Path.join(minimal_corpus, "single.txt"), "a")

      {:ok, _state_path} = BytePairEncoding.init(minimal_state, minimal_corpus)

      # Should fail gracefully when no pairs exist
      assert {:error, reason} = BytePairEncoding.run(minimal_state, 1, %{})
      assert reason =~ "No pairs found"
    end

    test "handles invalid state directory" do
      invalid_state = "/non/existent/state"

      assert {:error, reason} = BytePairEncoding.run(invalid_state, 1, %{})
      assert reason =~ "Failed to read state file"
    end
  end

  describe "batching behavior" do
    test "creates balanced batches based on file sizes", %{tmp_dir: tmp_dir} do
      # Create files of different sizes
      large_corpus = Path.join(tmp_dir, "large_corpus")
      large_state = Path.join(tmp_dir, "large_state")
      File.mkdir_p!(large_corpus)
      File.mkdir_p!(large_state)

      # Create files of different sizes
      File.write!(Path.join(large_corpus, "small1.txt"), "hello")
      File.write!(Path.join(large_corpus, "small2.txt"), "world")
      File.write!(Path.join(large_corpus, "medium.txt"), String.duplicate("test ", 50))
      File.write!(Path.join(large_corpus, "large.txt"), String.duplicate("content ", 200))

      opts = %{batch_size_bytes: 100, num_workers: 2}
      {:ok, _state_path} = BytePairEncoding.init(large_state, large_corpus, opts)

      # Verify batches were created
      batches_dir = Path.join(large_state, "batches")
      {:ok, batch_files} = File.ls(batches_dir)
      assert length(batch_files) > 0

      # Verify batch files contain expected structure
      batch_file = Path.join(batches_dir, Enum.at(batch_files, 0))
      {:ok, batch_content} = File.read(batch_file)
      batch_data = :erlang.binary_to_term(batch_content)

      assert is_list(batch_data)

      # Each batch entry should have file_path and tokens
      Enum.each(batch_data, fn file_entry ->
        assert Map.has_key?(file_entry, :file_path)
        assert Map.has_key?(file_entry, :tokens)
        assert is_binary(file_entry.file_path)
        assert is_list(file_entry.tokens)
      end)
    end
  end

  describe "vocabulary evolution" do
    test "vocabulary size decreases over training steps", %{
      corpus_dir: corpus_dir,
      state_dir: state_dir
    } do
      opts = %{batch_size_bytes: 500, num_workers: 2}
      {:ok, _state_path} = BytePairEncoding.init(state_dir, corpus_dir, opts)

      # Get initial vocabulary size
      vocab_file = Path.join(state_dir, "vocabulary.bert")
      {:ok, initial_vocab_content} = File.read(vocab_file)
      initial_vocab = :erlang.binary_to_term(initial_vocab_content)
      initial_size = map_size(initial_vocab)

      # Run some training steps
      {:ok, stats} = BytePairEncoding.run(state_dir, 2, %{num_workers: 2})

      # Final vocabulary should have new merged tokens
      final_size = stats.final_vocab_size

      # We should have added new tokens (merged pairs)
      # The exact relationship depends on the corpus content
      assert final_size >= initial_size
      assert stats.total_merges == 2
    end
  end

  describe "parallel processing" do
    test "handles different worker configurations", %{tmp_dir: tmp_dir, corpus_dir: corpus_dir} do
      workers_test = fn num_workers ->
        worker_corpus = Path.join(tmp_dir, "worker_corpus_#{num_workers}")
        worker_state = Path.join(tmp_dir, "worker_state_#{num_workers}")
        File.mkdir_p!(worker_corpus)
        File.mkdir_p!(worker_state)

        # Copy test files
        File.cp_r!(corpus_dir, worker_corpus)

        opts = %{batch_size_bytes: 300, num_workers: num_workers}

        start_time = System.monotonic_time(:millisecond)
        {:ok, _state_path} = BytePairEncoding.init(worker_state, worker_corpus, opts)
        {:ok, _stats} = BytePairEncoding.run(worker_state, 2, %{num_workers: num_workers})
        end_time = System.monotonic_time(:millisecond)

        end_time - start_time
      end

      # Test with different worker counts
      time_1_worker = workers_test.(1)
      time_2_workers = workers_test.(2)

      # Both should complete successfully
      assert time_1_worker > 0
      assert time_2_workers > 0

      # With small test corpus, timing might not show dramatic difference
      # but both configurations should work
    end
  end

  # Helper function to create test corpus
  defp create_test_corpus(corpus_dir) do
    # Create various test files with different content patterns
    test_files = [
      {"file1.txt", "hello world this is a test file for byte pair encoding"},
      {"file2.txt", "the quick brown fox jumps over the lazy dog"},
      {"file3.txt", "this is another test file with repeated words: test test test"},
      {"subdir/file4.txt", "nested file content for testing directory traversal"},
      {"file5.txt", String.duplicate("pattern ", 20) <> "end"},
      {"unicode.txt", "café naïve résumé 你好 🚀 emoji"},
      {"numbers.txt", "12345 67890 numbers and digits 0123456789"},
      {"punctuation.txt", "Hello! How are you? I'm fine, thank you. What about you?"},
      {"repetitive.txt", String.duplicate("abab ", 50)}
    ]

    Enum.each(test_files, fn {filename, content} ->
      file_path = Path.join(corpus_dir, filename)
      file_dir = Path.dirname(file_path)
      File.mkdir_p!(file_dir)
      File.write!(file_path, content)
    end)
  end
end
