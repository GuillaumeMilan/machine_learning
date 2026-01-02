#!/usr/bin/env elixir

# Example script for training a transformer language model with BPE tokenization
#
# Usage:
#   elixir scripts/train_transformer.exs

Mix.install([
  {:axon, "~> 0.6"},
  {:nx, "~> 0.7"},
  {:exla, "~> 0.7"},
  {:polaris, "~> 0.1"}
])

# Configure EXLA as the default backend
Nx.global_default_backend(EXLA.Backend)

alias MachineLearning.BytePairEncoding
alias MachineLearning.Tokenizer
alias MachineLearning.Transformer

defmodule TransformerTrainingExample do
  @moduledoc """
  Complete example of training a transformer language model.
  """

  def run do
    IO.puts("=== Transformer Language Model Training ===\n")

    # Step 1: Create or load tokenizer
    IO.puts("Step 1: Setting up tokenizer...")
    tokenizer = setup_tokenizer()
    IO.puts("Vocabulary size: #{Tokenizer.vocab_size(tokenizer)}\n")

    # Step 2: Prepare training data
    IO.puts("Step 2: Preparing training data...")
    {train_data, sample_texts} = prepare_training_data(tokenizer)
    IO.puts("Training data prepared.\n")

    # Step 3: Create transformer model
    IO.puts("Step 3: Creating transformer model...")

    model =
      Transformer.create_small_model(
        vocab_size: Tokenizer.vocab_size(tokenizer),
        max_seq_len: 128,
        embed_dim: 128,
        num_heads: 4,
        num_layers: 2,
        ff_dim: 512
      )

    IO.puts("Model created.\n")

    # Step 4: Initialize parameters
    IO.puts("Step 4: Initializing model parameters...")
    params = Transformer.init_params(model, seq_len: 64)
    IO.puts("Parameters initialized.\n")

    # Step 5: Train the model
    IO.puts("Step 5: Training model...")
    IO.puts("(This may take a while...)\n")

    trained_params =
      Transformer.train(model, params, train_data,
        epochs: 3,
        learning_rate: 0.0003
      )

    IO.puts("\nTraining completed!\n")

    # Step 6: Generate text
    IO.puts("Step 6: Generating text samples...")
    generate_samples(model, trained_params, tokenizer, sample_texts)

    IO.puts("\n=== Training Complete ===")
  end

  # Set up tokenizer from existing vocabulary or create new one
  defp setup_tokenizer do
    vocab_path = "vocabulary.bert"

    if File.exists?(vocab_path) do
      IO.puts("Loading existing vocabulary from #{vocab_path}...")
      Tokenizer.load(vocab_path)
    else
      IO.puts("Creating new vocabulary...")
      create_and_save_tokenizer(vocab_path)
    end
  end

  # Create a new tokenizer from corpus
  defp create_and_save_tokenizer(vocab_path) do
    corpus_dir = "./tmp/corpus"

    # Check if corpus exists
    if File.dir?(corpus_dir) do
      IO.puts("Compressing corpus to generate BPE vocabulary...")
      vocab = BytePairEncoding.compress(corpus_dir, 2000)

      tokenizer = Tokenizer.from_vocab(vocab)
      Tokenizer.save(tokenizer, vocab_path)

      IO.puts("Vocabulary saved to #{vocab_path}")
      tokenizer
    else
      IO.puts("Warning: No corpus found at #{corpus_dir}")
      IO.puts("Creating a minimal tokenizer with character-level vocabulary...")

      # Create a simple character-level tokenizer
      chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:\n"
      vocab = String.graphemes(chars) |> Enum.map(&BytePairEncoding.Token.new/1)

      tokenizer = Tokenizer.from_vocab(vocab)
      Tokenizer.save(tokenizer, vocab_path)

      tokenizer
    end
  end

  # Prepare training data from sample texts
  defp prepare_training_data(tokenizer) do
    # Sample training texts
    sample_texts = [
      "The quick brown fox jumps over the lazy dog.",
      "Machine learning is a fascinating field.",
      "Transformers have revolutionized natural language processing.",
      "Elixir is a functional programming language.",
      "Neural networks can learn complex patterns.",
      "Deep learning models require lots of data.",
      "Attention mechanisms are key to transformer success.",
      "Language models predict the next word in a sequence."
    ]

    # Encode texts to token sequences
    token_sequences =
      Enum.map(sample_texts, fn text ->
        Tokenizer.encode(tokenizer, text, add_special_tokens: true)
      end)

    # Prepare batched training data
    train_data =
      Transformer.prepare_training_data(token_sequences,
        batch_size: 4,
        seq_len: 32,
        shuffle: true
      )

    {train_data, sample_texts}
  end

  # Generate text samples using the trained model
  defp generate_samples(model, params, tokenizer, sample_prompts) do
    IO.puts("\nGenerating text from prompts:\n")

    Enum.take(sample_prompts, 3)
    |> Enum.each(fn prompt ->
      # Take first few words as prompt
      prompt_text = prompt |> String.split(" ") |> Enum.take(3) |> Enum.join(" ")

      IO.puts("Prompt: \"#{prompt_text}\"")

      # Encode prompt
      prompt_ids = Tokenizer.encode(tokenizer, prompt_text)
      prompt_tensor = Nx.tensor([prompt_ids])

      # Generate
      generated_ids =
        Transformer.generate(model, params, prompt_tensor,
          max_length: 20,
          temperature: 0.8,
          top_k: 10
        )

      # Decode
      generated_text = Tokenizer.decode(tokenizer, generated_ids)
      IO.puts("Generated: \"#{generated_text}\"\n")
    end)
  end
end

# Run the example
TransformerTrainingExample.run()
