defmodule MachineLearning.BytePairEncodingTest do
  use ExUnit.Case
  alias MachineLearning.BytePairEncoding
  alias MachineLearning.BytePairEncoding.Token

  @tmp_dir "test/tmp/bpe_test"

  setup do
    # Clean up before each test
    File.rm_rf!(@tmp_dir)
    File.mkdir_p!(@tmp_dir)

    on_exit(fn ->
      # Clean up after each test
      File.rm_rf!(@tmp_dir)
    end)

    :ok
  end

  describe "encode/3" do
    test "encodes the Wikipedia example 'aaabdaaabac' with vocab_size 3" do
      # Setup: Create a corpus file with the example text
      corpus_content = "aaabdaaabac"
      corpus_file = Path.join(@tmp_dir, "example.txt")
      File.write!(corpus_file, corpus_content)

      # Execute: Run byte pair encoding with vocab_size = 3
      tokens = BytePairEncoding.encode(@tmp_dir, 3, [])

      # Verify: Should have exactly 3 tokens
      assert length(tokens) == 3

      # The tokens should be in reverse order (most recent first)
      # Expected progression:
      # Step 1: "aa" appears 4 times (most frequent pair)
      # Step 2: "aaa" appears 2 times (aa + a)
      # Step 3: "aaab" appears 2 times (aaa + b)
      [token3, token2, token1] = tokens

      assert %Token{value: "aa"} = token1
      assert %Token{value: "aaa"} = token2
      assert %Token{value: "aaab"} = token3
    end

    test "encodes with larger corpus and custom vocab size" do
      # Create multiple files in the corpus
      File.write!(Path.join(@tmp_dir, "file1.txt"), "aaabdaaabac")
      File.write!(Path.join(@tmp_dir, "file2.txt"), "aaaaaa")

      tokens = BytePairEncoding.encode(@tmp_dir, 5, [])

      assert length(tokens) == 5
      # All tokens should be Token structs
      assert Enum.all?(tokens, &match?(%Token{}, &1))
    end
  end

  describe "tokenize/2" do
    test "tokenizes content using learned tokens" do
      content = "aaabdaaabac" |> String.graphemes()

      tokens = [
        Token.new("aaab")
      ]

      result = BytePairEncoding.tokenize(content, tokens)

      # Should replace the sequences with tokens
      assert Enum.any?(result, &match?(%Token{}, &1))
    end
  end
end
