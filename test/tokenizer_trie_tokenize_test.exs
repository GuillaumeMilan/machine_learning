defmodule MachineLearning.TokenizerTrieTokenizeTest do
  use ExUnit.Case, async: true
  alias MachineLearning.Tokenizer
  alias MachineLearning.BytePairEncoding.Token

  describe "tokenize/2 using vocab_map trie" do
    test "tokenizes simple single-character tokens" do
      tokens = [
        Token.new("a"),
        Token.new("b"),
        Token.new("c")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)
      graphemes = String.graphemes("abc")

      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 3
      assert Enum.all?(result, &match?(%Token{}, &1))
      assert Enum.map(result, & &1.value) == ["a", "b", "c"]
    end

    test "tokenizes with longest-match preference" do
      tokens = [
        Token.new("a"),
        Token.new("aa"),
        Token.new("aaa")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Should prefer "aaa" over "a" + "a" + "a"
      graphemes = String.graphemes("aaa")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 1
      assert [%Token{value: "aaa"}] = result
    end

    test "falls back to shorter tokens when longer don't match" do
      tokens = [
        Token.new("test"),
        Token.new("te"),
        Token.new("s")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # "tes" - should tokenize as "te" + "s"
      graphemes = String.graphemes("tes")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 2
      assert [%Token{value: "te"}, %Token{value: "s"}] = result
    end

    test "handles unknown characters as strings" do
      tokens = [
        Token.new("hello")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # "hellox" - should tokenize as "hello" + "x" (string)
      graphemes = String.graphemes("hellox")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 2
      assert [%Token{value: "hello"}, "x"] = result
    end

    test "tokenizes mixed known and unknown characters" do
      tokens = [
        Token.new("cat"),
        Token.new("dog"),
        # space as a token
        Token.new(" ")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # "cat dog!" - should tokenize as "cat" + " " + "dog" + "!"
      graphemes = String.graphemes("cat dog!")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 4
      assert [%Token{value: "cat"}, %Token{value: " "}, %Token{value: "dog"}, "!"] = result
    end

    test "handles complex overlapping prefixes correctly" do
      tokens = [
        Token.new("test"),
        Token.new("testing"),
        Token.new("tested"),
        Token.new("te")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Should prefer longest matches
      graphemes1 = String.graphemes("testing")
      result1 = Tokenizer.tokenize(graphemes1, tokenizer)
      assert [%Token{value: "testing"}] = result1

      graphemes2 = String.graphemes("tested")
      result2 = Tokenizer.tokenize(graphemes2, tokenizer)
      assert [%Token{value: "tested"}] = result2

      graphemes3 = String.graphemes("test")
      result3 = Tokenizer.tokenize(graphemes3, tokenizer)
      assert [%Token{value: "test"}] = result3
    end

    test "tokenizes with special tokens included" do
      tokens = [
        Token.new("hello"),
        Token.new("world")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: true)

      graphemes = String.graphemes("helloworld")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 2
      assert [%Token{value: "hello"}, %Token{value: "world"}] = result
    end

    test "handles empty input" do
      tokens = [Token.new("test")]
      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      result = Tokenizer.tokenize([], tokenizer)
      assert result == []
    end

    test "handles input with no matching tokens" do
      tokens = [Token.new("hello")]
      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      graphemes = String.graphemes("xyz")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 3
      assert ["x", "y", "z"] = result
    end

    test "tokenizes efficiently with many potential matches" do
      tokens = [
        Token.new("the"),
        Token.new("quick"),
        Token.new("brown"),
        Token.new("fox"),
        Token.new(" "),
        Token.new("t"),
        Token.new("he"),
        Token.new("qu"),
        Token.new("ick")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      graphemes = String.graphemes("the quick brown fox")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      # Should prefer longer tokens
      expected_values = ["the", " ", "quick", " ", "brown", " ", "fox"]

      actual_values =
        Enum.map(result, fn
          %Token{value: value} -> value
          string when is_binary(string) -> string
        end)

      assert actual_values == expected_values
    end

    test "handles unicode characters correctly" do
      tokens = [
        Token.new("🤖"),
        Token.new("hello"),
        Token.new("世界")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      graphemes = String.graphemes("hello🤖世界")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      assert length(result) == 3
      assert [%Token{value: "hello"}, %Token{value: "🤖"}, %Token{value: "世界"}] = result
    end

    test "handles partial matches at string boundaries" do
      tokens = [
        Token.new("testing"),
        Token.new("test")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # Input ends before completing "testing"
      graphemes = String.graphemes("test")
      result = Tokenizer.tokenize(graphemes, tokenizer)

      # Should get "test", not individual characters
      assert [%Token{value: "test"}] = result
    end
  end
end
