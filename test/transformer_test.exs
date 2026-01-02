defmodule MachineLearning.TransformerTest do
  use ExUnit.Case
  doctest MachineLearning.Transformer

  alias MachineLearning.Transformer
  alias MachineLearning.Tokenizer
  alias MachineLearning.BytePairEncoding.Token

  describe "create_model/1" do
    test "creates a transformer model with specified configuration" do
      model =
        Transformer.create_model(
          vocab_size: 1000,
          max_seq_len: 128,
          embed_dim: 64,
          num_heads: 4,
          num_layers: 2,
          ff_dim: 256
        )

      assert %Axon{} = model
    end

    test "creates a small model" do
      model = Transformer.create_small_model(vocab_size: 1000)
      assert %Axon{} = model
    end
  end

  describe "init_params/2" do
    test "initializes model parameters" do
      model = Transformer.create_small_model(vocab_size: 100)
      params = Transformer.init_params(model, seq_len: 32)

      assert is_map(params)
      assert map_size(params) > 0
    end
  end

  describe "prepare_training_data/2" do
    test "prepares token sequences into batched training data" do
      token_sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
      ]

      train_data =
        Transformer.prepare_training_data(
          token_sequences,
          batch_size: 2,
          seq_len: 4,
          shuffle: false
        )

      batches = Enum.to_list(train_data)
      assert length(batches) > 0

      first_batch = List.first(batches)
      assert %{input_ids: input_ids, labels: labels} = first_batch
      assert is_struct(input_ids, Nx.Tensor)
      assert is_struct(labels, Nx.Tensor)
    end
  end

  describe "Tokenizer" do
    setup do
      # Create a simple vocabulary
      vocab = [
        Token.new("a"),
        Token.new("b"),
        Token.new("c"),
        Token.new("ab"),
        Token.new("bc")
      ]

      tokenizer = Tokenizer.from_vocab(vocab)
      {:ok, tokenizer: tokenizer}
    end

    test "creates tokenizer from vocabulary", %{tokenizer: tokenizer} do
      assert %Tokenizer{} = tokenizer
      assert tokenizer.vocab_size > 0
    end

    test "encodes text to token IDs", %{tokenizer: tokenizer} do
      token_ids = Tokenizer.encode(tokenizer, "abc")
      assert is_list(token_ids)
      assert length(token_ids) > 0
    end

    test "decodes token IDs to text", %{tokenizer: tokenizer} do
      token_ids = Tokenizer.encode(tokenizer, "abc")
      text = Tokenizer.decode(tokenizer, token_ids)
      assert is_binary(text)
    end

    test "handles special tokens", %{tokenizer: tokenizer} do
      token_ids = Tokenizer.encode(tokenizer, "abc", add_special_tokens: true)
      # Should have BOS and EOS tokens
      assert length(token_ids) > 1
    end

    test "handles padding", %{tokenizer: tokenizer} do
      token_ids =
        Tokenizer.encode(
          tokenizer,
          "abc",
          max_length: 10,
          padding: true
        )

      assert length(token_ids) == 10
    end

    test "vocab_size returns correct size", %{tokenizer: tokenizer} do
      size = Tokenizer.vocab_size(tokenizer)
      assert is_integer(size)
      assert size > 0
    end
  end

  describe "save and load tokenizer" do
    test "saves and loads tokenizer correctly" do
      vocab = [Token.new("a"), Token.new("b")]
      tokenizer = Tokenizer.from_vocab(vocab)

      path = "test_vocab.bert"

      try do
        Tokenizer.save(tokenizer, path)
        loaded_tokenizer = Tokenizer.load(path)

        assert Tokenizer.vocab_size(loaded_tokenizer) == Tokenizer.vocab_size(tokenizer)
      after
        File.rm(path)
      end
    end
  end

  describe "integration test" do
    @tag :integration
    @tag timeout: 120_000
    test "trains a small transformer model" do
      # Create a minimal tokenizer
      vocab = Enum.map(0..50, fn i -> Token.new("tok_#{i}") end)
      tokenizer = Tokenizer.from_vocab(vocab)

      # Create simple training data
      token_sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 10]
      ]

      train_data =
        Transformer.prepare_training_data(
          token_sequences,
          batch_size: 2,
          seq_len: 4,
          shuffle: false
        )

      # Create a tiny model
      model =
        Transformer.create_model(
          vocab_size: Tokenizer.vocab_size(tokenizer),
          max_seq_len: 32,
          embed_dim: 32,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 64,
          dropout_rate: 0.0
        )

      # Initialize params
      params = Transformer.init_params(model, seq_len: 4)

      # Train for 1 epoch (just to verify it works)
      trained_params =
        Transformer.train(
          model,
          params,
          train_data,
          epochs: 1,
          learning_rate: 0.001
        )

      assert is_map(trained_params)
      assert map_size(trained_params) > 0
    end
  end

  describe "generate with untrained model" do
    @tag timeout: 120_000
    test "generates text from untrained model using vocabulary.bert" do
      vocab_path = "vocabulary.bert"

      # Skip test if vocabulary.bert doesn't exist
      unless File.exists?(vocab_path) do
        IO.puts("Skipping test: #{vocab_path} not found")
        assert true
      else
        # Load vocabulary tokens as shown in the documentation
        tokens =
          File.read!(vocab_path)
          |> :erlang.binary_to_term()
          |> Enum.map(&MachineLearning.BytePairEncoding.Token.new/1)

        # Create tokenizer from tokens
        tokenizer = Tokenizer.from_vocab(tokens)
        vocab_size = Tokenizer.vocab_size(tokenizer)

        IO.puts("Loaded vocabulary with size: #{vocab_size}")

        # Create a small model
        model =
          Transformer.create_small_model(
            vocab_size: vocab_size,
            max_seq_len: 64
          )

        # Initialize parameters (without training)
        params = Transformer.init_params(model, seq_len: 32)

        # Create a prompt
        prompt = "The future of AI"
        prompt_ids = Tokenizer.encode(tokenizer, prompt)
        prompt_tensor = Nx.tensor([prompt_ids])

        IO.puts("Prompt: #{inspect(prompt)}")
        IO.puts("Prompt IDs: #{inspect(prompt_ids)}")
        IO.puts("Prompt tensor shape: #{inspect(Nx.shape(prompt_tensor))}")

        # Generate text (this should work even without training)
        generated_ids =
          Transformer.generate(
            model,
            params,
            prompt_tensor,
            max_length: 20,
            temperature: 0.8,
            top_k: 10
          )

        # Verify we got output
        assert is_struct(generated_ids, Nx.Tensor)

        # Decode the generated text
        generated_text = Tokenizer.decode(tokenizer, generated_ids)

        IO.puts("Generated text: #{inspect(generated_text)}")

        # Verify we have some output
        assert is_binary(generated_text)
        assert String.length(generated_text) > 0
      end
    end

    test "generates text with simple dummy vocabulary" do
      # Create a minimal tokenizer
      vocab = Enum.map(0..99, fn i -> Token.new("tok_#{i}") end)
      tokenizer = Tokenizer.from_vocab(vocab)

      # Create model
      model =
        Transformer.create_model(
          vocab_size: Tokenizer.vocab_size(tokenizer),
          max_seq_len: 32,
          embed_dim: 32,
          num_heads: 2,
          num_layers: 1,
          ff_dim: 64,
          dropout_rate: 0.0
        )

      # Initialize params
      params = Transformer.init_params(model, seq_len: 16)

      # Create prompt tensor
      prompt_ids = [5, 10, 15]
      prompt_tensor = Nx.tensor([prompt_ids])

      # Generate
      generated_ids =
        Transformer.generate(
          model,
          params,
          prompt_tensor,
          max_length: 10,
          temperature: 1.0,
          top_k: 10
        )

      assert is_struct(generated_ids, Nx.Tensor)

      # Verify shape - should be {1, max_length}
      {batch, seq_len} = Nx.shape(generated_ids)
      assert batch == 1
      assert seq_len == 10

      # Decode and verify
      generated_text = Tokenizer.decode(tokenizer, generated_ids)
      assert is_binary(generated_text)
    end
  end
end
