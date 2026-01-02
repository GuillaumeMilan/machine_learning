# LLM Training with BPE Tokenization - Quick Start Guide

This guide shows how to train a transformer language model using byte-pair encoding tokenization.

## Step 1: Generate BPE Vocabulary

First, create a corpus and generate a BPE vocabulary:

```elixir
# Add files to corpus
MachineLearning.BytePairEncoding.add_to_corpus(
  "./lib",  # Source directory (e.g., your Elixir code)
  "./tmp/corpus",
  extensions: [".ex", ".exs"]
)

# Generate BPE vocabulary with target size
vocab = MachineLearning.BytePairEncoding.compress(
  "./tmp/corpus",
  5000  # Target vocabulary size
)

# Save vocabulary
File.write!("vocabulary.bert", :erlang.term_to_binary(vocab))

# Optional: View corpus statistics
MachineLearning.Corpus.stats("./tmp/corpus")
```

## Step 2: Load Vocabulary and Create Tokenizer

Load the saved vocabulary and create a tokenizer:

```elixir
# Load vocabulary tokens
tokens = File.read!("vocabulary.bert") 
  |> :erlang.binary_to_term() 
  |> Enum.map(&MachineLearning.BytePairEncoding.Token.new/1)

# Create tokenizer from tokens
tokenizer = MachineLearning.Tokenizer.from_vocab(tokens)

# Save tokenizer for later use
MachineLearning.Tokenizer.save(tokenizer, "tokenizer.bert")

# Or load existing tokenizer
tokenizer = MachineLearning.Tokenizer.load("tokenizer.bert")
```

## Step 3: Prepare Training Data

Tokenize your training texts:

```elixir
# Option 1: Load texts from your corpus directory
training_texts = MachineLearning.Corpus.load_sample("./tmp/corpus", 1000)

# Option 2: Load and split by lines (good for code/structured text)
training_texts = MachineLearning.Corpus.load_split_texts(
  "./tmp/corpus",
  split_by: :lines,
  min_length: 50,
  max_files: 100
)

# Option 3: Load and chunk into fixed-size pieces
training_texts = MachineLearning.Corpus.load_chunked_texts(
  "./tmp/corpus",
  chunk_size: 500,
  overlap: 50,
  max_files: 100
)

# Option 4: Manual list of texts
training_texts = [
  "The quick brown fox jumps over the lazy dog.",
  "Machine learning with Elixir is powerful.",
  "Transformers revolutionized natural language processing.",
]

# Encode texts to token IDs
token_sequences = Enum.map(training_texts, fn text ->
  MachineLearning.Tokenizer.encode(tokenizer, text, add_special_tokens: true)
end)

# Prepare batched training data
train_data = MachineLearning.Transformer.prepare_training_data(
  token_sequences,
  batch_size: 32,
  seq_len: 128,
  shuffle: true
)
```

## Step 4: Create and Train Model

```elixir
# Create transformer model
model = MachineLearning.Transformer.create_model(
  vocab_size: MachineLearning.Tokenizer.vocab_size(tokenizer),
  max_seq_len: 512,
  embed_dim: 256,
  num_heads: 8,
  num_layers: 6,
  ff_dim: 1024,
  dropout_rate: 0.1
)

# For faster experimentation, use the small model
model = MachineLearning.Transformer.create_small_model(
  vocab_size: MachineLearning.Tokenizer.vocab_size(tokenizer)
)

# Initialize model parameters
params = MachineLearning.Transformer.init_params(model, seq_len: 128)

# Train the model
trained_params = MachineLearning.Transformer.train(
  model,
  params,
  train_data,
  epochs: 10,
  learning_rate: 0.0001,
  optimizer: :adamw
)
```

## Step 5: Generate Text

```elixir
# Encode a prompt
prompt = "The future of AI"
prompt_ids = MachineLearning.Tokenizer.encode(tokenizer, prompt)
prompt_tensor = Nx.tensor([prompt_ids])

# Generate text
generated_ids = MachineLearning.Transformer.generate(
  model,
  trained_params,
  prompt_tensor,
  max_length: 50,
  temperature: 0.8,
  top_k: 40
)

# Decode generated tokens to text
generated_text = MachineLearning.Tokenizer.decode(tokenizer, generated_ids)
IO.puts("Generated: #{generated_text}")
```

## Complete Example in IEx

```elixir
# Start IEx with EXLA backend
iex -S mix

# Configure EXLA
Nx.global_default_backend(EXLA.Backend)

# Load tokenizer
tokenizer = MachineLearning.Tokenizer.load("vocabulary.bert")

# Or if you have raw tokens:
tokens = File.read!("vocabulary.bert") 
  |> :erlang.binary_to_term() 
  |> Enum.map(&MachineLearning.BytePairEncoding.Token.new/1)
tokenizer = MachineLearning.Tokenizer.from_vocab(tokens)

# Prepare some training data
# Load from corpus
texts = MachineLearning.Corpus.load_sample("./tmp/corpus", 50)

# Or use manual texts for testing
# texts = [
#   "Elixir is a functional programming language.",
#   "Machine learning models can learn patterns.",
#   "Transformers use attention mechanisms.",
# ]

token_sequences = Enum.map(texts, fn text ->
  MachineLearning.Tokenizer.encode(tokenizer, text, add_special_tokens: true)
end)

train_data = MachineLearning.Transformer.prepare_training_data(
  token_sequences,
  batch_size: 2,
  seq_len: 32
)

# Create small model for quick testing
model = MachineLearning.Transformer.create_small_model(
  vocab_size: MachineLearning.Tokenizer.vocab_size(tokenizer)
)

params = MachineLearning.Transformer.init_params(model, seq_len: 32)

# Train for 1 epoch
trained_params = MachineLearning.Transformer.train(
  model,
  params,
  train_data,
  epochs: 1
)

# Generate
prompt_ids = MachineLearning.Tokenizer.encode(tokenizer, "Elixir is")
generated = MachineLearning.Transformer.generate(
  model,
  trained_params,
  Nx.tensor([prompt_ids]),
  max_length: 20
)
MachineLearning.Tokenizer.decode(tokenizer, generated)
```

## Running the Example Script

A complete training example is provided in `scripts/train_transformer.exs`:

```bash
elixir scripts/train_transformer.exs
```

## Tips

1. **Start Small**: Use `create_small_model/1` and small datasets for initial experiments
2. **Vocabulary Size**: 2,000-10,000 tokens works well for most applications
3. **Training**: Monitor the loss - if it doesn't decrease, try adjusting the learning rate
4. **Generation**: Lower temperature (0.5-0.8) produces more coherent text
5. **EXLA Backend**: Always use `Nx.global_default_backend(EXLA.Backend)` for better performance

## Troubleshooting

- **Out of Memory**: Reduce `batch_size`, `seq_len`, or use `create_small_model/1`
- **Slow Training**: Make sure EXLA backend is enabled
- **Poor Quality**: Train longer, use more data, or increase model size
