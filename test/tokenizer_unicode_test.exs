defmodule MachineLearning.TokenizerUnicodeTest do
  use ExUnit.Case, async: true
  alias MachineLearning.Tokenizer
  alias MachineLearning.BytePairEncoding.Token

  describe "decode/3 with unicode sanitization" do
    test "handles valid UTF-8 characters correctly" do
      # Create a simple tokenizer with valid UTF-8 tokens
      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>"),
        Token.new("Hello"),
        Token.new(" world"),
        Token.new("!")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Decode valid token IDs
      result = Tokenizer.decode(tokenizer, [4, 5, 6])

      assert result == "Hello world!"
    end

    test "sanitizes invalid UTF-8 byte sequences" do
      # Create tokens with invalid UTF-8 sequences
      # These are raw bytes that don't form valid UTF-8
      # Invalid UTF-8
      invalid_utf8_bytes = <<255, 254, 253>>

      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>"),
        Token.new("valid"),
        Token.new(invalid_utf8_bytes),
        Token.new("text")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Decode with invalid bytes in the middle
      result = Tokenizer.decode(tokenizer, [4, 5, 6])

      # Should not crash and should replace invalid bytes
      assert is_binary(result)
      assert String.valid?(result)
      # The invalid bytes should be replaced with '?'
      assert result == "valid???text"
    end

    test "filters out control characters except common whitespace" do
      # Create tokens with control characters
      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>"),
        Token.new("Hello"),
        # NULL character (should be filtered)
        Token.new(<<0>>),
        # SOH character (should be filtered)
        Token.new(<<1>>),
        # Space (should be kept)
        Token.new(" "),
        # Newline (should be kept)
        Token.new("\n"),
        # Tab (should be kept)
        Token.new("\t"),
        Token.new("world")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Decode with control characters
      result = Tokenizer.decode(tokenizer, [4, 5, 6, 7, 8, 9, 10])

      # Control characters (NULL, SOH) should be filtered out
      # Common whitespace (space, newline, tab) should be kept
      assert result == "Hello \n\tworld"
    end

    test "handles tokens with mixed valid and invalid Unicode" do
      # Simulate what happens when model generates invalid token IDs
      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>"),
        Token.new("The"),
        Token.new(" future"),
        Token.new(" of"),
        Token.new(" AI"),
        # Invalid UTF-8 sequences that might come from model generation
        # Incomplete UTF-8
        Token.new(<<194, 191>>),
        # Valid Thai character
        Token.new(<<224, 184, 138>>),
        # Invalid UTF-8
        Token.new(<<250, 195>>),
        Token.new("is")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Decode a sequence with mixed valid/invalid
      result = Tokenizer.decode(tokenizer, [4, 5, 6, 7, 8, 9, 10, 11])

      # Should handle it gracefully without crashing
      assert is_binary(result)
      assert String.valid?(result)
      # Should contain the valid parts
      assert result =~ "The future of AI"
    end

    test "preserves valid extended Unicode characters" do
      # Test with various Unicode ranges
      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>"),
        Token.new("Hello"),
        Token.new(" "),
        # Chinese characters
        Token.new("世界"),
        Token.new(" "),
        # Emoji
        Token.new("🌍"),
        Token.new(" "),
        # Greek letter
        Token.new("Ω")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      result = Tokenizer.decode(tokenizer, [4, 5, 6, 7, 8, 9, 10])

      # All valid Unicode should be preserved
      assert result == "Hello 世界 🌍 Ω"
    end

    test "handles empty token sequences" do
      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      result = Tokenizer.decode(tokenizer, [])

      assert result == ""
    end

    test "handles unknown token IDs gracefully" do
      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>"),
        Token.new("valid")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Use token IDs that don't exist (999, 1000)
      result = Tokenizer.decode(tokenizer, [4, 999, 1000])

      # Should default to <UNK> for unknown IDs and handle gracefully
      assert is_binary(result)
      assert String.valid?(result)
    end

    test "skip_special_tokens option works with sanitization" do
      tokens = [
        Token.new("<PAD>"),
        Token.new("<UNK>"),
        Token.new("<BOS>"),
        Token.new("<EOS>"),
        Token.new("Hello"),
        Token.new(" world")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # With special tokens
      result_with = Tokenizer.decode(tokenizer, [0, 1, 2, 3, 4, 5], skip_special_tokens: false)
      assert result_with == "<PAD><UNK><BOS><EOS>Hello world"

      # Without special tokens (default)
      result_without = Tokenizer.decode(tokenizer, [0, 1, 2, 3, 4, 5], skip_special_tokens: true)
      assert result_without == "Hello world"
    end
  end
end
