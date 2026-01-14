defmodule MachineLearning.TokenizerVerificationTest do
  use ExUnit.Case, async: true

  alias MachineLearning.Tokenizer
  alias MachineLearning.BytePairEncoding.Token
  @moduletag :tmp_dir

  setup %{tmp_dir: tmp_dir} do
    # Create a test tokenizer with known vocabulary
    test_vocab = [
      Token.new("h"),
      Token.new("e"),
      Token.new("l"),
      Token.new("o"),
      Token.new(" "),
      Token.new("w"),
      Token.new("r"),
      Token.new("d"),
      Token.new("!"),
      Token.new("hello"),
      Token.new("world"),
      Token.new("he"),
      Token.new("ll"),
      Token.new("lo"),
      Token.new("or")
    ]

    tokenizer = Tokenizer.from_vocab(test_vocab, add_special_tokens: true)

    %{tokenizer: tokenizer, tmp_dir: tmp_dir}
  end

  describe "verify/3" do
    test "analyzes corpus with all tokens used", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create test files with content that uses all our vocabulary
      content1 = "hello world!"
      content2 = "he world hello"

      file1 = Path.join(test_dir, "file1.txt")
      file2 = Path.join(test_dir, "file2.txt")
      File.write!(file1, content1)
      File.write!(file2, content2)

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir)

      assert stats.total_files == 2
      assert stats.total_characters == String.length(content1) + String.length(content2)
      assert stats.total_tokens > 0
      # No unknown tokens with this simple example
      assert stats.unk_token_count == 0
      # All tokens covered
      assert stats.coverage_percentage == 100.0
      # Some vocab used
      assert stats.vocab_utilization_percentage > 0.0

      # Should have used_tokens and unused_tokens sets
      assert is_struct(stats.used_tokens, MapSet)
      assert is_struct(stats.unused_tokens, MapSet)
      assert stats.used_vocab_tokens + stats.unused_vocab_tokens == tokenizer.vocab_size
    end

    test "handles unknown characters correctly", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create content with characters not in our vocabulary
      # 'x', 'y', 'z' are not in our vocab
      content = "hello xyz world"

      file1 = Path.join(test_dir, "unknown.txt")
      File.write!(file1, content)

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir)

      assert stats.total_files == 1
      # Should have unknown tokens for x, y, z
      assert stats.unk_token_count > 0
      # Not all content could be tokenized properly
      assert stats.coverage_percentage < 100.0
      # Should have sample sequences
      assert length(stats.sample_unk_sequences) > 0
    end

    test "respects max_files option", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create multiple files
      for i <- 1..5 do
        file = Path.join(test_dir, "file#{i}.txt")
        File.write!(file, "hello world #{i}")
      end

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir, max_files: 3)

      # Should only process 3 files
      assert stats.total_files == 3
    end

    test "filters by file extensions", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create files with different extensions
      File.write!(Path.join(test_dir, "text.txt"), "hello world")
      File.write!(Path.join(test_dir, "markdown.md"), "hello world")
      File.write!(Path.join(test_dir, "readme.txt"), "hello world")

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir, file_extensions: [".txt"])

      # Only .txt files should be processed
      assert stats.total_files == 2
    end

    test "handles nested directories", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create nested directory structure
      subdir = Path.join(test_dir, "subdir")
      File.mkdir_p!(subdir)

      File.write!(Path.join(test_dir, "root.txt"), "hello")
      File.write!(Path.join(subdir, "nested.txt"), "world")

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir)

      # Should find files in subdirectories
      assert stats.total_files == 2
      assert stats.total_characters == String.length("hello") + String.length("world")
    end

    test "handles empty corpus directory", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir)

      assert stats.total_files == 0
      assert stats.total_characters == 0
      assert stats.total_tokens == 0
      assert stats.unk_token_count == 0
      assert stats.coverage_percentage == 0.0
      assert MapSet.size(stats.used_tokens) == 0
      assert MapSet.size(stats.unused_tokens) == tokenizer.vocab_size
    end

    test "handles non-existent directory", %{tokenizer: tokenizer} do
      non_existent_dir = "/path/that/does/not/exist"

      {:error, reason} = Tokenizer.verify(tokenizer, non_existent_dir)

      assert String.contains?(reason, "does not exist")
    end

    test "calculates vocabulary utilization correctly", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create content that uses only a subset of vocabulary
      # Only uses 'h', 'e', 'l', 'o', and potentially 'hello' tokens
      content = "hello"

      file = Path.join(test_dir, "simple.txt")
      File.write!(file, content)

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir)

      # Should have some tokens used, but not all
      assert stats.used_vocab_tokens > 0
      assert stats.unused_vocab_tokens > 0
      assert stats.used_vocab_tokens < tokenizer.vocab_size
      assert stats.vocab_utilization_percentage > 0.0
      assert stats.vocab_utilization_percentage < 100.0
    end

    test "limits UNK sequence samples", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create content with many unknown characters
      # Many unknown sequences
      content = String.duplicate("xyz", 20)

      file = Path.join(test_dir, "unknown.txt")
      File.write!(file, content)

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir, sample_size: 5)

      # Should be limited to sample_size
      assert length(stats.sample_unk_sequences) <= 5
    end

    test "handles files that cannot be read", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      # Create a valid file and a directory (which will fail to read as text)
      File.write!(Path.join(test_dir, "valid.txt"), "hello")
      # Dir with .txt extension
      File.mkdir_p!(Path.join(test_dir, "directory.txt"))

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir)

      # Should process what it can and skip what it can't
      # Should not crash
      assert stats.total_files >= 0
    end

    test "provides detailed token analysis", %{tokenizer: tokenizer, tmp_dir: test_dir} do
      content = "hello world"

      file = Path.join(test_dir, "analysis.txt")
      File.write!(file, content)

      {:ok, stats} = Tokenizer.verify(tokenizer, test_dir)

      # Verify all expected fields are present
      expected_fields = [
        :total_files,
        :total_characters,
        :total_tokens,
        :used_vocab_tokens,
        :unused_vocab_tokens,
        :unk_token_count,
        :coverage_percentage,
        :vocab_utilization_percentage,
        :used_tokens,
        :unused_tokens,
        :sample_unk_sequences
      ]

      for field <- expected_fields do
        assert Map.has_key?(stats, field), "Missing field: #{field}"
      end

      # Verify percentage calculations are reasonable
      assert stats.coverage_percentage >= 0.0 and stats.coverage_percentage <= 100.0

      assert stats.vocab_utilization_percentage >= 0.0 and
               stats.vocab_utilization_percentage <= 100.0
    end
  end
end
