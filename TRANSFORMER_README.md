# Transformer Language Model

A decoder-only transformer architecture implementation in Elixir using Axon, designed for autoregressive language modeling.

## Overview

This implementation provides a complete pipeline for training language models:

1. **Byte-Pair Encoding (BPE)** tokenization for efficient vocabulary
2. **Transformer Decoder** architecture with multi-head attention
3. **Training utilities** for language modeling tasks
4. **Text generation** with temperature and top-k sampling

## Architecture

The transformer decoder consists of:

- **Token Embeddings**: Maps token IDs to dense vectors
- **Positional Embeddings**: Learned positional encodings
- **Multi-Head Self-Attention**: Causal (masked) attention layers
- **Feed-Forward Networks**: Position-wise FFN with GELU activation
- **Layer Normalization**: Pre-normalization architecture
- **Residual Connections**: Skip connections around each sub-layer

### Model Sizes

**Small Model** (for experimentation):
- Embedding dimension: 128
- Attention heads: 4
- Layers: 2
- FFN dimension: 512
- Parameters: ~few million

**Full Model** (for production):
- Embedding dimension: 256
- Attention heads: 8
- Layers: 6
- FFN dimension: 1024
- Parameters: ~tens of millions

## Quick Start

### 1. Prepare Your Corpus

First, add text files to your corpus:

```elixir
# Add source code or text files to corpus
MachineLearning.BytePairEncoding.add_to_corpus(
  "./source_directory",
  "./tmp/corpus",
  extensions: [".ex", ".md", ".txt"]
)
```

### 2. Create Vocabulary

Generate a BPE vocabulary from your corpus:

```elixir
# Compress corpus to create vocabulary
vocab = MachineLearning.BytePairEncoding.compress(
  "./tmp/corpus",
  5000  # target vocabulary size
)

# Create and save tokenizer
tokenizer = MachineLearning.Tokenizer.from_vocab(vocab)
MachineLearning.Tokenizer.save(tokenizer, "vocabulary.bert")
```

### 3. Prepare Training Data

Tokenize your training texts:

```elixir
# Load tokenizer
tokenizer = MachineLearning.Tokenizer.load("vocabulary.bert")

# Encode your texts
texts = [
  "The quick brown fox jumps over the lazy dog.",
  "Machine learning is fascinating.",
  # ... more texts
]

token_sequences = Enum.map(texts, fn text ->
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

### 4. Create and Train Model

```elixir
# Create model
model = MachineLearning.Transformer.create_model(
  vocab_size: MachineLearning.Tokenizer.vocab_size(tokenizer),
  max_seq_len: 512,
  embed_dim: 256,
  num_heads: 8,
  num_layers: 6,
  ff_dim: 1024,
  dropout_rate: 0.1
)

# Initialize parameters
params = MachineLearning.Transformer.init_params(model, seq_len: 128)

# Train
trained_params = MachineLearning.Transformer.train(
  model,
  params,
  train_data,
  epochs: 10,
  learning_rate: 0.0001,
  optimizer: :adamw
)
```

### 5. Generate Text

```elixir
# Encode a prompt
prompt = "The future of AI is"
prompt_ids = MachineLearning.Tokenizer.encode(tokenizer, prompt)
prompt_tensor = Nx.tensor([prompt_ids])

# Generate continuation
generated_ids = MachineLearning.Transformer.generate(
  model,
  trained_params,
  prompt_tensor,
  max_length: 100,
  temperature: 0.8,
  top_k: 50
)

# Decode to text
generated_text = MachineLearning.Tokenizer.decode(tokenizer, generated_ids)
IO.puts("Generated: #{generated_text}")
```

## Running the Example Script

A complete training example is provided:

```bash
# Make sure you have compiled the project
mix compile

# Run the training script
elixir scripts/train_transformer.exs
```

Or from IEx:

```elixir
# Start IEx with the project
iex -S mix

# Load the script
Code.require_file("scripts/train_transformer.exs")
```

## API Reference

### Transformer Module

- `create_model/1` - Create a transformer decoder model
- `create_small_model/1` - Create a smaller model for experiments
- `init_params/2` - Initialize model parameters
- `train/4` - Train the model on text data
- `generate/4` - Generate text autoregressively
- `evaluate/4` - Evaluate model on test data
- `prepare_training_data/2` - Prepare tokenized sequences for training

### Tokenizer Module

- `from_vocab/2` - Create tokenizer from BPE vocabulary
- `load/1` - Load tokenizer from file
- `save/2` - Save tokenizer to file
- `encode/3` - Encode text to token IDs
- `encode_batch/3` - Encode multiple texts
- `decode/3` - Decode token IDs to text
- `decode_batch/3` - Decode multiple sequences
- `vocab_size/1` - Get vocabulary size

## Configuration Options

### Model Configuration

```elixir
create_model(
  vocab_size: 5000,        # Size of vocabulary (required)
  max_seq_len: 512,        # Maximum sequence length
  embed_dim: 256,          # Embedding dimension
  num_heads: 8,            # Number of attention heads
  num_layers: 6,           # Number of transformer layers
  ff_dim: 1024,            # Feed-forward network dimension
  dropout_rate: 0.1        # Dropout rate
)
```

### Training Configuration

```elixir
train(model, params, data,
  epochs: 10,              # Number of training epochs
  learning_rate: 0.0001,   # Learning rate
  optimizer: :adamw        # Optimizer (:adamw, :adam, :sgd)
)
```

### Generation Configuration

```elixir
generate(model, params, prompt,
  max_length: 100,         # Maximum tokens to generate
  temperature: 0.8,        # Sampling temperature (higher = more random)
  top_k: 50                # Top-k sampling
)
```

## Best Practices

### Data Preparation

1. **Corpus Size**: Use at least 10MB of text for meaningful results
2. **Vocabulary Size**: 2,000-10,000 tokens for most applications
3. **Sequence Length**: 128-512 tokens balances quality and speed
4. **Batch Size**: Larger batches (32-64) for stable training

### Training

1. **Start Small**: Use `create_small_model/1` for initial experiments
2. **Learning Rate**: Start with 0.0001 and adjust based on loss curves
3. **Optimizer**: AdamW generally works best for transformers
4. **Epochs**: Monitor loss; stop when it plateaus

### Generation

1. **Temperature**: Lower (0.5-0.8) for coherent text, higher (1.0-1.5) for creative
2. **Top-k**: 10-50 for focused sampling
3. **Prompts**: Longer prompts (5-10 tokens) give better context

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce `seq_len`
- Use `create_small_model/1`
- Reduce `num_layers`

### Slow Training

- Enable EXLA backend: `Nx.global_default_backend(EXLA.Backend)`
- Reduce model size
- Reduce sequence length
- Use smaller batch size

### Poor Generation Quality

- Train for more epochs
- Increase model size
- Use larger/better corpus
- Adjust temperature and top-k

## Technical Details

### Causal Masking

The transformer uses causal (autoregressive) masking to ensure tokens can only attend to previous positions, preventing information leakage from future tokens during training.

### Loss Function

Uses sparse categorical cross-entropy loss, which is efficient for large vocabularies and doesn't require one-hot encoding of labels.

### Positional Encoding

Uses learned positional embeddings rather than fixed sinusoidal encodings, allowing the model to learn position-dependent patterns.

## Future Enhancements

Potential improvements:

- [ ] Flash attention for efficiency
- [ ] Rotary positional embeddings (RoPE)
- [ ] Multi-query attention
- [ ] KV cache for faster generation
- [ ] Gradient checkpointing for larger models
- [ ] Mixed precision training
- [ ] Distributed training support
- [ ] Model checkpointing and resumption

## References

- Vaswani et al. (2017) - "Attention Is All You Need"
- Radford et al. (2018) - "Improving Language Understanding by Generative Pre-Training" (GPT)
- Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)

## License

Same as the parent project.
