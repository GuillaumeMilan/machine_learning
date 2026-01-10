defmodule MachineLearning.TokenizerTest do
  use ExUnit.Case, async: true
  alias MachineLearning.Tokenizer
  alias MachineLearning.BytePairEncoding.Token

  describe "from_vocab/2 with vocab_map building" do
    test "builds vocab_map from simple vocabulary" do
      tokens = [
        Token.new("hello"),
        Token.new("world")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      assert is_map(tokenizer.vocab_map)
      assert Map.has_key?(tokenizer.vocab_map, "h")
      assert Map.has_key?(tokenizer.vocab_map, "w")
    end

    test "creates nested structure for tokens sharing prefixes" do
      tokens = [
        Token.new("a"),
        Token.new("aa"),
        Token.new("aaa")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      vocab_map = tokenizer.vocab_map

      # Check first level: "a"
      assert vocab_map["a"]["is_token?"] == true
      assert is_map(vocab_map["a"]["children"])

      # Check second level: "aa"
      children_a = vocab_map["a"]["children"]
      assert children_a["a"]["is_token?"] == true
      assert is_map(children_a["a"]["children"])

      # Check third level: "aaa"
      children_aa = children_a["a"]["children"]
      assert children_aa["a"]["is_token?"] == true
      assert children_aa["a"]["children"] == %{}
    end

    test "correctly marks tokens and non-tokens" do
      tokens = [
        Token.new("cat"),
        Token.new("car")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      vocab_map = tokenizer.vocab_map

      # "c" should not be a token by itself
      assert vocab_map["c"]["is_token?"] == false

      # "ca" should not be a token by itself
      ca_node = vocab_map["c"]["children"]["a"]
      assert ca_node["is_token?"] == false

      # "cat" should be a token
      cat_node = ca_node["children"]["t"]
      assert cat_node["is_token?"] == true

      # "car" should be a token
      car_node = ca_node["children"]["r"]
      assert car_node["is_token?"] == true
    end

    test "includes special tokens in vocab_map when add_special_tokens is true" do
      tokens = [Token.new("hello")]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: true)

      vocab_map = tokenizer.vocab_map

      # Special tokens should be present
      assert Map.has_key?(vocab_map, "<")

      # Navigate to <PAD>
      pad_node =
        vocab_map["<"]["children"]["P"]["children"]["A"]["children"]["D"]["children"][">"]

      assert pad_node["is_token?"] == true

      # Regular token should also be there
      assert Map.has_key?(vocab_map, "h")
    end

    test "handles single character tokens" do
      tokens = [
        Token.new("a"),
        Token.new("b"),
        Token.new("c")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      vocab_map = tokenizer.vocab_map

      assert vocab_map["a"]["is_token?"] == true
      assert vocab_map["b"]["is_token?"] == true
      assert vocab_map["c"]["is_token?"] == true

      # Each should have empty children
      assert vocab_map["a"]["children"] == %{}
      assert vocab_map["b"]["children"] == %{}
      assert vocab_map["c"]["children"] == %{}
    end

    test "handles complex prefix structures" do
      tokens = [
        Token.new("test"),
        Token.new("testing"),
        Token.new("tested")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      vocab_map = tokenizer.vocab_map

      # Navigate to "test"
      test_path = vocab_map["t"]["children"]["e"]["children"]["s"]["children"]["t"]
      assert test_path["is_token?"] == true

      # Navigate to "testing"
      children_test = test_path["children"]
      assert children_test["i"]["children"]["n"]["children"]["g"]["is_token?"] == true

      # Navigate to "tested"
      assert children_test["e"]["children"]["d"]["is_token?"] == true
    end

    test "vocab_map persists across tokenizer creation" do
      tokens = [
        Token.new("hello"),
        Token.new("world")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      # vocab_map should be stable
      vocab_map_1 = tokenizer.vocab_map
      vocab_map_2 = tokenizer.vocab_map

      assert vocab_map_1 == vocab_map_2
    end

    test "vocab_map structure is correct for multi-branch tokens" do
      tokens = [
        Token.new("ab"),
        Token.new("ac"),
        Token.new("ad")
      ]

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      vocab_map = tokenizer.vocab_map

      # First level: "a"
      a_node = vocab_map["a"]
      assert a_node["is_token?"] == false
      assert is_map(a_node["children"])

      # Second level: multiple branches
      children_a = a_node["children"]
      assert Map.has_key?(children_a, "b")
      assert Map.has_key?(children_a, "c")
      assert Map.has_key?(children_a, "d")

      # All should be tokens
      assert children_a["b"]["is_token?"] == true
      assert children_a["c"]["is_token?"] == true
      assert children_a["d"]["is_token?"] == true
    end

    test "empty vocabulary produces empty vocab_map" do
      tokens = []

      tokenizer = Tokenizer.from_vocab(tokens, add_special_tokens: false)

      assert tokenizer.vocab_map == %{}
    end

    test "special tokens only produces special token nodes" do
      tokenizer = Tokenizer.from_vocab([], add_special_tokens: true)

      vocab_map = tokenizer.vocab_map

      # Should have at least the first character of special tokens
      assert Map.has_key?(vocab_map, "<")
    end
  end
end
