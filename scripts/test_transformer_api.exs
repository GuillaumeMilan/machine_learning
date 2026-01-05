#!/usr/bin/env elixir

# Test script for the new Transformer API
# Usage: mix run scripts/test_transformer_api.exs

IO.puts("=== Testing MachineLearning.Transformer API ===\n")

# Set EXLA backend for performance
Nx.global_default_backend(EXLA.Backend)

# Clean up any existing test model
test_model_dir = "test/tmp/test_transformer_api"
if File.dir?(test_model_dir), do: File.rm_rf!(test_model_dir)

# Step 1: Create dummy vocabulary tokens
IO.puts("Step 1: Creating dummy vocabulary...")

# Create a simple vocabulary with some common tokens
dummy_tokens = [
  # Special tokens
  %{token: "[PAD]", id: 0},
  %{token: "[UNK]", id: 1},
  %{token: "[BOS]", id: 2},
  %{token: "[EOS]", id: 3},
  # Common words
  %{token: "the", id: 4},
  %{token: "quick", id: 5},
  %{token: "brown", id: 6},
  %{token: "fox", id: 7},
  %{token: "jumps", id: 8},
  %{token: "over", id: 9},
  %{token: "lazy", id: 10},
  %{token: "dog", id: 11},
  %{token: "in", id: 12},
  %{token: "a", id: 13},
  %{token: "machine", id: 14},
  %{token: "learning", id: 15},
  %{token: "is", id: 16},
  %{token: "fun", id: 17},
  %{token: "and", id: 18},
  %{token: "powerful", id: 19},
  # Add more tokens to reach ~100
  %{token: "transformer", id: 20},
  %{token: "model", id: 21},
  %{token: "training", id: 22},
  %{token: "data", id: 23},
  %{token: "neural", id: 24},
  %{token: "network", id: 25}
]

# Pad to 100 tokens
dummy_tokens =
  dummy_tokens ++
    Enum.map(26..99, fn i ->
      %{token: "token_#{i}", id: i}
    end)

IO.puts("Created #{length(dummy_tokens)} tokens\n")

# Step 2: Create a new model
IO.puts("Step 2: Creating new transformer model...")

try do
  model =
    MachineLearning.Transformer.create(
      tokens: dummy_tokens,
      save_dir: test_model_dir,
      vocab_size: 100,
      # Small for testing
      max_seq_len: 64,
      # Small for testing
      embed_dim: 64,
      # Small for testing
      num_heads: 4,
      # Small for testing
      num_layers: 2,
      # Small for testing
      ff_dim: 256
    )

  IO.puts("\n✅ Model created successfully!")
  IO.inspect(model.config, label: "Model config")
rescue
  e ->
    IO.puts("\n❌ Error creating model:")
    IO.inspect(e)
    System.halt(1)
end

# Step 3: Create dummy training data
IO.puts("\n\nStep 3: Creating dummy training data...")

# Create some simple token sequences
dummy_sequences = [
  # "the quick brown fox jumps over the lazy dog"
  [2, 4, 5, 6, 7, 8, 9, 4, 10, 11, 3],
  # "machine learning is fun and powerful"
  [2, 14, 15, 16, 17, 18, 19, 3],
  # "transformer model training is fun"
  [2, 20, 21, 22, 16, 17, 3],
  # "neural network learning is powerful"
  [2, 24, 25, 15, 16, 19, 3],
  # Repeat a few times to have more data
  [2, 4, 5, 6, 7, 8, 9, 4, 10, 11, 3],
  [2, 14, 15, 16, 17, 18, 19, 3],
  [2, 20, 21, 22, 16, 17, 3],
  [2, 24, 25, 15, 16, 19, 3]
]

IO.puts("Created #{length(dummy_sequences)} token sequences")

# Step 4: Add training data to model
IO.puts("\n\nStep 4: Adding training data to model...")

try do
  MachineLearning.Transformer.add_training_data(
    test_model_dir,
    "test_dataset",
    token_sequences: dummy_sequences,
    # Small batch for testing
    batch_size: 2,
    # Small sequence for testing
    seq_len: 16
  )

  IO.puts("\n✅ Training data added successfully!")
rescue
  e ->
    IO.puts("\n❌ Error adding training data:")
    IO.inspect(e)
    System.halt(1)
end

# Step 5: List available resources
IO.puts("\n\nStep 5: Listing available resources...")

params = MachineLearning.Transformer.list_params(test_model_dir)
IO.puts("Available parameters: #{inspect(params)}")

datasets = MachineLearning.Transformer.list_training_data(test_model_dir)
IO.puts("Available datasets: #{inspect(datasets)}")

info = MachineLearning.Transformer.info(test_model_dir)
IO.puts("\nModel info:")
IO.inspect(info, pretty: true)

# Step 6: Train the model (just 2 epochs for testing)
IO.puts("\n\nStep 6: Training model (2 epochs)...")

try do
  trained_model =
    MachineLearning.Transformer.train(
      test_model_dir,
      "test_dataset",
      epochs: 2,
      learning_rate: 0.001,
      params_version: "initial"
    )

  IO.puts("\n✅ Training completed successfully!")
rescue
  e ->
    IO.puts("\n❌ Error during training:")
    IO.inspect(e)
    System.halt(1)
end

# Step 7: Check saved parameters
IO.puts("\n\nStep 7: Checking saved parameters...")

params_after_training = MachineLearning.Transformer.list_params(test_model_dir)
IO.puts("Parameters after training: #{inspect(params_after_training)}")

# Step 8: Load model with different parameter versions
IO.puts("\n\nStep 8: Testing model loading with different parameter versions...")

try do
  model_epoch_1 = MachineLearning.Transformer.load(test_model_dir, params_version: "epoch_001")
  IO.puts("✓ Loaded model with epoch_001 parameters")

  model_epoch_2 = MachineLearning.Transformer.load(test_model_dir, params_version: "epoch_002")
  IO.puts("✓ Loaded model with epoch_002 parameters")

  model_latest = MachineLearning.Transformer.load(test_model_dir)
  IO.puts("✓ Loaded model with latest parameters")
rescue
  e ->
    IO.puts("\n❌ Error loading model:")
    IO.inspect(e)
    System.halt(1)
end

# Step 9: Test prediction
IO.puts("\n\nStep 9: Testing text generation...")

try do
  # Use a simple prompt that matches our vocabulary
  generated_text =
    MachineLearning.Transformer.predict(
      test_model_dir,
      "the quick",
      max_length: 20,
      temperature: 1.0
    )

  IO.puts("\nPrompt: \"the quick\"")
  IO.puts("Generated: \"#{generated_text}\"")
  IO.puts("\n✅ Prediction successful!")
rescue
  e ->
    IO.puts("\n❌ Error during prediction:")
    IO.inspect(e)
    System.halt(1)
end

# Step 10: Test loading and reusing training data
IO.puts("\n\nStep 10: Testing loading existing training data...")

try do
  {loaded_data, metadata} =
    MachineLearning.Transformer.load_training_data(
      test_model_dir,
      "test_dataset"
    )

  batch_count = Enum.count(loaded_data)
  IO.puts("✓ Loaded #{batch_count} batches")
  IO.puts("Metadata: #{inspect(metadata)}")

  IO.puts("\n✅ Training data loading successful!")
rescue
  e ->
    IO.puts("\n❌ Error loading training data:")
    IO.inspect(e)
    System.halt(1)
end

# Step 11: Continue training from latest checkpoint
IO.puts("\n\nStep 11: Continuing training from latest checkpoint (2 more epochs)...")

try do
  continued_model =
    MachineLearning.Transformer.train(
      test_model_dir,
      "test_dataset",
      epochs: 2,
      # Lower learning rate for fine-tuning
      learning_rate: 0.0005,
      params_version: "latest"
    )

  IO.puts("\n✅ Continued training successful!")

  final_params = MachineLearning.Transformer.list_params(test_model_dir)
  IO.puts("Final parameters: #{inspect(final_params)}")
rescue
  e ->
    IO.puts("\n❌ Error during continued training:")
    IO.inspect(e)
    System.halt(1)
end

IO.puts("\n\n" <> String.duplicate("=", 50))
IO.puts("✅ ALL TESTS PASSED!")
IO.puts(String.duplicate("=", 50))

IO.puts("\nTest model saved at: #{test_model_dir}")
IO.puts("You can inspect the folder structure to see the organization.")

# Optionally clean up
# IO.puts("\nCleaning up test model...")
# File.rm_rf!(test_model_dir)
# IO.puts("✓ Cleanup complete")
