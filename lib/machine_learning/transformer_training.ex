defmodule MachineLearning.TransformerTraining do
  @moduledoc """
  Complete example of training a transformer language model.
  """
  alias MachineLearning.Tokenizer
  alias MachineLearning.Transformer

  @doc """
  Load a trained model and tokenizer from a directory.

  ## Parameters

  - `model_dir`: Path to the directory containing saved model files

  ## Returns

  A tuple `{model, params, tokenizer}` containing the loaded model, trained parameters, and tokenizer.
  The model configuration is automatically loaded from the saved `config.json` file.

  ## Examples

      iex> {model, params, tokenizer} = MachineLearning.TransformerTraining.load("models/transformer_1735862400")
      iex> # Generate text
      iex> prompt_ids = Tokenizer.encode(tokenizer, "The quick brown")
      iex> prompt_tensor = Nx.tensor([prompt_ids])
      iex> generated_ids = Transformer.generate(model, params, prompt_tensor, max_length: 20)
      iex> generated_text = Tokenizer.decode(tokenizer, generated_ids)
  """
  def load(model_dir) do
    IO.puts("Loading model from #{model_dir}...")

    # Load model configuration
    config_path = Path.join(model_dir, "config.json")

    unless File.exists?(config_path) do
      raise "Model configuration file not found: #{config_path}"
    end

    config =
      File.read!(config_path)
      |> Jason.decode!()

    IO.puts("Loaded model configuration")

    # Load tokenizer
    tokenizer_path = Path.join(model_dir, "tokenizer.bert")

    unless File.exists?(tokenizer_path) do
      raise "Tokenizer file not found: #{tokenizer_path}"
    end

    tokenizer = Tokenizer.load(tokenizer_path)
    IO.puts("Loaded tokenizer with vocabulary size: #{Tokenizer.vocab_size(tokenizer)}")

    # Load parameters
    params_path = Path.join(model_dir, "params.bin")

    unless File.exists?(params_path) do
      raise "Parameters file not found: #{params_path}"
    end

    params =
      File.read!(params_path)
      |> :erlang.binary_to_term()

    IO.puts("Loaded trained parameters")

    # Create model with the saved architecture
    model =
      Transformer.create_model(
        vocab_size: Tokenizer.vocab_size(tokenizer),
        max_seq_len: config["max_seq_len"],
        embed_dim: config["embed_dim"],
        num_heads: config["num_heads"],
        num_layers: config["num_layers"],
        ff_dim: config["ff_dim"] || 512,
        dropout_rate: config["dropout_rate"] || 0.1
      )

    IO.puts(
      "Created model with saved architecture: embed_dim=#{config["embed_dim"]}, num_heads=#{config["num_heads"]}, num_layers=#{config["num_layers"]}"
    )

    IO.puts("Model loaded successfully from #{model_dir}\n")

    {model, params, tokenizer}
  end

  @doc """
  Generate text using a loaded model.

  ## Parameters

  - `model_tuple`: A tuple `{model, params, tokenizer}` returned by `load/1`
  - `prompt_text`: The text prompt to start generation from
  - `opts`: Generation options (same as `Transformer.generate/4`)

  ## Examples

      iex> model_data = MachineLearning.TransformerTraining.load("models/transformer_1735862400")
      iex> MachineLearning.TransformerTraining.predict(model_data, "The quick brown")
      "The quick brown fox jumps over..."
  """
  def predict({model, params, tokenizer}, prompt_text, opts \\ []) do
    prompt_ids = Tokenizer.encode(tokenizer, prompt_text)
    prompt_tensor = Nx.tensor([prompt_ids])

    generated_ids =
      Transformer.generate(model, params, prompt_tensor,
        max_length: Keyword.get(opts, :max_length, 50),
        temperature: Keyword.get(opts, :temperature, 0.9),
        top_k: Keyword.get(opts, :top_k, 40),
        top_p: Keyword.get(opts, :top_p, 0.92),
        repetition_penalty: Keyword.get(opts, :repetition_penalty, 1.3),
        no_repeat_ngram_size: Keyword.get(opts, :no_repeat_ngram_size, 3)
      )

    Tokenizer.decode(tokenizer, generated_ids)
  end

  def run(config \\ %{}) do
    IO.puts("=== Transformer Language Model Training ===\n")

    # Step 1: Create or load tokenizer
    IO.puts("Step 1: Setting up tokenizer...")
    tokenizer = setup_tokenizer(config)
    IO.puts("Vocabulary size: #{Tokenizer.vocab_size(tokenizer)}\n")

    # Step 2: Prepare training data
    IO.puts("Step 2: Preparing training data...")
    {train_data, sample_texts, _all_texts} = prepare_training_data(tokenizer, config)
    IO.puts("Training data prepared.\n")

    # Step 3: Create transformer model with improved architecture
    IO.puts("Step 3: Creating transformer model...")

    # Get model config with better defaults
    # Increased from 128
    embed_dim = Map.get(config, :embed_dim, 256)
    # Increased from 4
    num_heads = Map.get(config, :num_heads, 8)
    # Increased from 2
    num_layers = Map.get(config, :num_layers, 4)
    # Increased from 512
    ff_dim = Map.get(config, :ff_dim, 1024)
    # Increased from 128
    max_seq_len = Map.get(config, :max_seq_len, 256)

    model =
      Transformer.create_model(
        vocab_size: Tokenizer.vocab_size(tokenizer),
        max_seq_len: max_seq_len,
        embed_dim: embed_dim,
        num_heads: num_heads,
        num_layers: num_layers,
        ff_dim: ff_dim,
        dropout_rate: 0.1
      )

    IO.puts(
      "Model created with: embed_dim=#{embed_dim}, heads=#{num_heads}, layers=#{num_layers}\n"
    )

    # Step 4: Initialize parameters
    IO.puts("Step 4: Initializing model parameters...")
    # Increased from 64
    seq_len = Map.get(config, :seq_len, 128)
    params = Transformer.init_params(model, seq_len: seq_len)
    IO.puts("Parameters initialized.\n")

    # Step 5: Train the model with improved hyperparameters
    IO.puts("Step 5: Training model...")
    IO.puts("(This may take a while...)\n")

    # Increased default from 3
    epoch = Map.get(config, :epoch, 10)
    # Increased from 0.0003
    learning_rate = Map.get(config, :learning_rate, 0.001)

    trained_params =
      Transformer.train(model, params, train_data,
        epochs: epoch,
        learning_rate: learning_rate
      )

    IO.puts("\nTraining completed!\n")

    # Step 6: Generate text
    IO.puts("Step 6: Generating text samples...")
    generate_samples(model, trained_params, tokenizer, sample_texts)

    IO.puts("\n=== Training Complete ===")

    # Step 7: Save tokenizer and trained parameters
    IO.puts("Step 7: Saving model and tokenizer...")
    save_dir = save_model_and_tokenizer(tokenizer, trained_params, config)
    IO.puts("Saved to directory: #{save_dir}\n")

    IO.puts("\n=== Training Complete ===")
    {model, trained_params, tokenizer}
  end

  def run_on_model(model_dir, config \\ %{}) do
    {model, params, tokenizer} = load(model_dir)
    {train_data, sample_texts, _all_texts} = prepare_training_data(tokenizer, config)
    epoch = Map.get(config, :epoch, 10)
    learning_rate = Map.get(config, :learning_rate, 0.001)

    trained_params =
      Transformer.train(model, params, train_data,
        epochs: epoch,
        learning_rate: learning_rate
      )

    generate_samples(model, trained_params, tokenizer, sample_texts)
    save_dir = save_model_and_tokenizer(tokenizer, trained_params, config)
    IO.puts("Saved to directory: #{save_dir}\n")

    IO.puts("\n=== Training Complete ===")
    {model, trained_params, tokenizer}
  end

  # Set up tokenizer from existing vocabulary or create new one
  defp setup_tokenizer(config) do
    vocab_path = Map.get(config, :vocab_path, "vocabulary.bert")

    if File.exists?(vocab_path) do
      IO.puts("Loading existing vocabulary from #{vocab_path}...")

      tokens =
        File.read!("vocabulary.bert")
        |> :erlang.binary_to_term()
        |> Enum.map(&MachineLearning.BytePairEncoding.Token.new/1)

      MachineLearning.Tokenizer.from_vocab(tokens)
    else
      raise "Vocabulary file not found: #{vocab_path}. Please create a vocabulary before running training."
    end
  end

  # Prepare training data from sample texts
  defp prepare_training_data(tokenizer, config) do
    corpus_dir = Map.get(config, :corpus_dir, nil)
    sample_size = Map.get(config, :sample_size, nil)

    # Try to load from corpus, otherwise use sample texts
    all_texts =
      if corpus_dir do
        if not File.dir?(corpus_dir) do
          raise "Corpus directory not found: #{corpus_dir}"
        end

        IO.puts("Loading all texts from corpus directory...")
        MachineLearning.Corpus.load_texts(corpus_dir)
      else
        IO.puts("Corpus not found, using sample texts...")

        [
          "The quick brown fox jumps over the lazy dog in the beautiful forest.",
          "Machine learning is a fascinating field with many practical applications.",
          "Transformers have revolutionized natural language processing and understanding.",
          "Elixir is a functional programming language built on the Erlang VM.",
          "Neural networks can learn complex patterns from large amounts of data.",
          "Deep learning models require lots of data and computational resources.",
          "Attention mechanisms are key to transformer architecture success and efficiency.",
          "Language models predict the next word in a sequence using context.",
          "Byte pair encoding creates efficient vocabularies for text tokenization.",
          "Training deep models requires significant compute resources and time management."
        ]
      end

    IO.puts("Loaded #{length(all_texts)} total texts from corpus")

    # Sample from the corpus if sample_size is specified
    sample_texts =
      if sample_size && sample_size < length(all_texts) do
        IO.puts("Sampling #{sample_size} texts for training...")
        Enum.take_random(all_texts, sample_size)
      else
        all_texts
      end

    IO.puts("Using #{length(sample_texts)} texts for training")

    # Encode texts to token sequences
    token_sequences =
      sample_texts
      |> Stream.chunk_every(10)
      |> Task.async_stream(
        fn chunk ->
          start = System.monotonic_time(:millisecond)

          Enum.map(chunk, fn text ->
            Tokenizer.encode(tokenizer, text, add_special_tokens: true)
          end)
          |> tap(fn _ ->
            text_size = chunk |> Enum.map(&String.length/1) |> Enum.sum()
            duration = System.monotonic_time(:millisecond) - start
            IO.puts("Tokenized text of length #{text_size} in #{duration} ms")
          end)
        end,
        timeout: :infinity
      )
      |> Enum.flat_map(fn {:ok, result} -> result end)

    IO.puts("Tokenized training data with #{Enum.count(token_sequences)} sequences.")
    # Increased from 4 for more stable gradients
    batch_size = Map.get(config, :batch_size, 16)
    # Increased from 32 for better context
    seq_len = Map.get(config, :seq_len, 128)

    # Prepare batched training data
    train_data =
      Transformer.prepare_training_data(token_sequences,
        batch_size: batch_size,
        seq_len: seq_len,
        shuffle: true
      )

    IO.puts("Prepared training data with #{Enum.count(train_data)} batches.")

    {train_data, sample_texts, all_texts}
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

      # Generate with anti-repetition settings
      generated_ids =
        Transformer.generate(model, params, prompt_tensor,
          max_length: 50,
          # Higher for more diversity
          temperature: 0.9,
          # Broader sampling
          top_k: 40,
          # Nucleus sampling
          top_p: 0.92,
          # Penalize repetition
          repetition_penalty: 1.3,
          # Block 3-gram repetition
          no_repeat_ngram_size: 3
        )

      # Decode
      generated_text = Tokenizer.decode(tokenizer, generated_ids)
      IO.puts("Generated: \"#{generated_text}\"\n")
    end)
  end

  # Save tokenizer and trained parameters to a directory
  defp save_model_and_tokenizer(tokenizer, params, config) do
    # Create save directory with timestamp
    timestamp = DateTime.utc_now() |> DateTime.to_unix()
    save_dir = Map.get(config, :save_dir, "models/transformer_#{timestamp}")

    # Create directory if it doesn't exist
    File.mkdir_p!(save_dir)

    # Save model configuration with actual values used
    model_config = %{
      "max_seq_len" => Map.get(config, :max_seq_len, 256),
      "embed_dim" => Map.get(config, :embed_dim, 256),
      "num_heads" => Map.get(config, :num_heads, 8),
      "num_layers" => Map.get(config, :num_layers, 4),
      "ff_dim" => Map.get(config, :ff_dim, 1024),
      "vocab_size" => Tokenizer.vocab_size(tokenizer),
      "saved_at" => DateTime.utc_now() |> DateTime.to_iso8601()
    }

    config_path = Path.join(save_dir, "config.json")
    File.write!(config_path, Jason.encode!(model_config, pretty: true))

    # Save tokenizer
    tokenizer_path = Path.join(save_dir, "tokenizer.bert")
    Tokenizer.save(tokenizer, tokenizer_path)

    # Save trained parameters
    params_path = Path.join(save_dir, "params.bin")
    binary_params = :erlang.term_to_binary(params)
    File.write!(params_path, binary_params)

    IO.puts("Model and tokenizer saved successfully in: #{save_dir}")

    save_dir
  end
end
