defmodule MachineLearning.Transformer.Backend do
  @moduledoc """
  Transformer decoder architecture for language modeling using Axon.

  This module provides a simplified decoder-only transformer architecture suitable for
  autoregressive language generation.

  **Note:** This is a simplified implementation. The attention mechanism uses
  dense layers instead of full multi-head attention for compatibility.
  For production use, consider implementing proper multi-head attention with causal masking.

  ## Example Usage

      # Create a transformer model
      model = MachineLearning.Transformer.Backend.create_model(
        vocab_size: 5000,
        max_seq_len: 512,
        embed_dim: 256,
        num_heads: 8,
        num_layers: 6,
        ff_dim: 1024
      )

      # Initialize parameters
      params = MachineLearning.Transformer.Backend.init_params(model)

      # Train the model
      trained_params = MachineLearning.Transformer.Backend.train(
        model,
        params,
        train_data,
        epochs: 10
      )

      # Generate text
      MachineLearning.Transformer.Backend.generate(
        model,
        trained_params,
        prompt_tokens,
        max_length: 100
      )
  """

  @doc """
  Creates a transformer decoder model for language modeling.

  ## Parameters

  - `opts`: Model configuration options
    - `:vocab_size` - Size of the vocabulary (required)
    - `:max_seq_len` - Maximum sequence length (default: 512)
    - `:embed_dim` - Embedding dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_layers` - Number of transformer layers (default: 6)
    - `:ff_dim` - Feed-forward network dimension (default: 1024)
    - `:dropout_rate` - Dropout rate (default: 0.1)

  ## Examples

      iex> model = MachineLearning.Transformer.Backend.create_model(vocab_size: 5000)
      iex> is_struct(model, Axon)
      true
  """
  @spec create_model(keyword()) :: Axon.t()
  def create_model(opts) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    max_seq_len = Keyword.get(opts, :max_seq_len, 512)
    embed_dim = Keyword.get(opts, :embed_dim, 256)
    num_heads = Keyword.get(opts, :num_heads, 8)
    num_layers = Keyword.get(opts, :num_layers, 6)
    ff_dim = Keyword.get(opts, :ff_dim, 1024)
    dropout_rate = Keyword.get(opts, :dropout_rate, 0.1)

    # Input: {batch_size, seq_len} of token IDs
    Axon.input("input_ids", shape: {nil, nil})
    |> token_and_position_embedding(vocab_size, max_seq_len, embed_dim)
    |> then(fn x ->
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        transformer_decoder_layer(acc, embed_dim, num_heads, ff_dim, dropout_rate, layer_idx)
      end)
    end)
    |> Axon.layer_norm(name: "final_norm")
    # Output projection to vocabulary
    |> Axon.dense(vocab_size, name: "output_projection")
  end

  @doc """
  Creates a simpler transformer model for faster experimentation.

  ## Examples

      iex> model = MachineLearning.Transformer.Backend.create_small_model(vocab_size: 5000)
      iex> is_struct(model, Axon)
      true
  """
  @spec create_small_model(keyword()) :: Axon.t()
  def create_small_model(opts) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    max_seq_len = Keyword.get(opts, :max_seq_len, 256)

    create_model(
      vocab_size: vocab_size,
      max_seq_len: max_seq_len,
      embed_dim: 128,
      num_heads: 4,
      num_layers: 2,
      ff_dim: 512,
      dropout_rate: 0.1
    )
  end

  # Token and positional embedding layer
  defp token_and_position_embedding(input, vocab_size, max_seq_len, embed_dim) do
    # Token embedding
    token_embed =
      input
      |> Axon.embedding(vocab_size, embed_dim, name: "token_embedding")

    # Positional embedding - learned positions
    pos_embed =
      Axon.input("position_ids", shape: {nil, nil})
      |> Axon.embedding(max_seq_len, embed_dim, name: "position_embedding")

    # Combine token and position embeddings
    Axon.add([token_embed, pos_embed], name: "embeddings")
  end

  # Single transformer decoder layer
  defp transformer_decoder_layer(input, embed_dim, num_heads, ff_dim, dropout_rate, layer_idx) do
    # Multi-head self-attention block
    attention_output =
      input
      |> causal_self_attention(embed_dim, num_heads, dropout_rate, layer_idx)
      |> Axon.dropout(rate: dropout_rate, name: "attention_dropout_#{layer_idx}")

    # Add & Norm (residual connection + layer norm)
    norm1 =
      Axon.add([input, attention_output], name: "residual_1_#{layer_idx}")
      |> Axon.layer_norm(name: "norm_1_#{layer_idx}")

    # Feed-forward network
    ff_output =
      norm1
      |> Axon.dense(ff_dim, activation: :gelu, name: "ff_1_#{layer_idx}")
      |> Axon.dropout(rate: dropout_rate, name: "ff_dropout_#{layer_idx}")
      |> Axon.dense(embed_dim, name: "ff_2_#{layer_idx}")
      |> Axon.dropout(rate: dropout_rate, name: "ff_dropout_2_#{layer_idx}")

    # Add & Norm
    Axon.add([norm1, ff_output], name: "residual_2_#{layer_idx}")
    |> Axon.layer_norm(name: "norm_2_#{layer_idx}")
  end

  # Causal (masked) self-attention mechanism
  defp causal_self_attention(input, embed_dim, num_heads, dropout_rate, layer_idx) do
    # Proper multi-head attention with causal masking
    head_dim = div(embed_dim, num_heads)

    # Query, Key, Value projections
    query = Axon.dense(input, embed_dim, name: "query_#{layer_idx}")
    key = Axon.dense(input, embed_dim, name: "key_#{layer_idx}")
    value = Axon.dense(input, embed_dim, name: "value_#{layer_idx}")

    # Reshape for multi-head attention: {batch, seq_len, num_heads, head_dim}
    query = reshape_for_attention(query, num_heads, head_dim, "query_reshape_#{layer_idx}")
    key = reshape_for_attention(key, num_heads, head_dim, "key_reshape_#{layer_idx}")
    value = reshape_for_attention(value, num_heads, head_dim, "value_reshape_#{layer_idx}")

    # Scaled dot-product attention with causal mask
    attention_output =
      scaled_dot_product_attention(
        query,
        key,
        value,
        head_dim,
        dropout_rate,
        layer_idx
      )

    # Reshape back: {batch, seq_len, embed_dim}
    attention_output =
      reshape_from_attention(attention_output, embed_dim, "attention_reshape_#{layer_idx}")

    # Output projection
    Axon.dense(attention_output, embed_dim, name: "attention_output_#{layer_idx}")
  end

  # Reshape tensor for multi-head attention
  defp reshape_for_attention(tensor, num_heads, head_dim, name) do
    Axon.layer(
      fn x, _opts ->
        {batch, seq_len, _embed_dim} = Nx.shape(x)

        x
        |> Nx.reshape({batch, seq_len, num_heads, head_dim})
        # {batch, num_heads, seq_len, head_dim}
        |> Nx.transpose(axes: [0, 2, 1, 3])
      end,
      [tensor],
      name: name,
      op_name: :reshape_for_attention
    )
  end

  # Reshape tensor back from multi-head attention
  defp reshape_from_attention(tensor, embed_dim, name) do
    Axon.layer(
      fn x, _opts ->
        {batch, _num_heads, seq_len, _head_dim} = Nx.shape(x)

        x
        # {batch, seq_len, num_heads, head_dim}
        |> Nx.transpose(axes: [0, 2, 1, 3])
        |> Nx.reshape({batch, seq_len, embed_dim})
      end,
      [tensor],
      name: name,
      op_name: :reshape_from_attention
    )
  end

  # Scaled dot-product attention with causal masking
  defp scaled_dot_product_attention(query, key, value, head_dim, dropout_rate, layer_idx) do
    # Compute attention scores: Q * K^T / sqrt(head_dim)
    scores =
      Axon.layer(
        fn q, k, _opts ->
          # q, k shapes: {batch, num_heads, seq_len, head_dim}
          {batch, num_heads, seq_len, _head_dim} = Nx.shape(q)

          # Reshape to combine batch and head dimensions
          # {batch * num_heads, seq_len, head_dim}
          q_reshaped = Nx.reshape(q, {batch * num_heads, seq_len, head_dim})
          k_reshaped = Nx.reshape(k, {batch * num_heads, seq_len, head_dim})

          # Transpose k: {batch * num_heads, head_dim, seq_len}
          k_t = Nx.transpose(k_reshaped, axes: [0, 2, 1])

          # Matrix multiplication with batch dimension
          # q_reshaped: {batch * num_heads, seq_len, head_dim}
          # k_t:        {batch * num_heads, head_dim, seq_len}
          # Contract on head_dim (axis 2 of q, axis 1 of k_t)
          # Keep batch dimension (axis 0)
          scores_flat = Nx.dot(q_reshaped, [2], [0], k_t, [1], [0])

          # Reshape back: {batch, num_heads, seq_len, seq_len}
          scores = Nx.reshape(scores_flat, {batch, num_heads, seq_len, seq_len})

          # Scale by sqrt(head_dim)
          scale = :math.sqrt(head_dim)
          Nx.divide(scores, scale)
        end,
        [query, key],
        name: "attention_scores_#{layer_idx}",
        op_name: :compute_attention_scores
      )

    # Apply causal mask
    masked_scores =
      Axon.layer(
        fn scores, _opts ->
          {batch, num_heads, seq_len, _} = Nx.shape(scores)

          # Create causal mask: lower triangular matrix
          # Use Nx.iota to create efficient mask
          row_indices = Nx.iota({seq_len, 1})
          col_indices = Nx.iota({1, seq_len})

          # mask[i, j] = 0 if j <= i, else -inf
          # This allows position i to attend to positions 0..i
          mask =
            Nx.select(
              Nx.greater(col_indices, row_indices),
              Nx.Constants.neg_infinity(),
              0.0
            )

          # Broadcast mask to match scores shape and add
          mask = Nx.broadcast(mask, {batch, num_heads, seq_len, seq_len})
          Nx.add(scores, mask)
        end,
        [scores],
        name: "causal_mask_#{layer_idx}",
        op_name: :apply_causal_mask
      )

    # Apply softmax
    attention_weights =
      Axon.softmax(masked_scores, axis: -1, name: "attention_softmax_#{layer_idx}")

    # Apply dropout
    attention_weights =
      Axon.dropout(attention_weights,
        rate: dropout_rate,
        name: "attention_weights_dropout_#{layer_idx}"
      )

    # Apply attention to values: attention_weights * V
    Axon.layer(
      fn weights, v, _opts ->
        # weights: {batch, num_heads, seq_len, seq_len}
        # v: {batch, num_heads, seq_len, head_dim}
        {batch, num_heads, seq_len, _} = Nx.shape(weights)
        {_, _, _, head_dim} = Nx.shape(v)

        # Reshape to combine batch and head dimensions
        weights_flat = Nx.reshape(weights, {batch * num_heads, seq_len, seq_len})
        v_flat = Nx.reshape(v, {batch * num_heads, seq_len, head_dim})

        # Matrix multiplication with batch dimension
        # weights_flat: {batch * num_heads, seq_len, seq_len}
        # v_flat:       {batch * num_heads, seq_len, head_dim}
        # Contract on seq_len (axis 2 of weights, axis 1 of v)
        result_flat = Nx.dot(weights_flat, [2], [0], v_flat, [1], [0])

        # Reshape back: {batch, num_heads, seq_len, head_dim}
        Nx.reshape(result_flat, {batch, num_heads, seq_len, head_dim})
      end,
      [attention_weights, value],
      name: "attention_apply_#{layer_idx}",
      op_name: :apply_attention
    )
  end

  @doc """
  Initializes transformer model parameters.

  ## Parameters

  - `model`: The Axon transformer model
  - `seq_len`: Sequence length for initialization (default: 128)
  """
  @spec init_params(Axon.t(), keyword()) :: map()
  def init_params(model, opts \\ []) do
    seq_len = Keyword.get(opts, :seq_len, 128)

    # Create templates for initialization
    input_template = Nx.template({1, seq_len}, :s64)
    position_template = Nx.template({1, seq_len}, :s64)

    {init_fn, _predict_fn} = Axon.build(model)

    init_fn.(
      %{
        "input_ids" => input_template,
        "position_ids" => position_template
      },
      Axon.ModelState.new(%{})
    )
  end

  @doc """
  Trains the transformer model on text data.

  ## Parameters

  - `model`: The Axon transformer model
  - `params`: Initial model parameters
  - `train_data`: Training dataset (stream of token sequences)
  - `opts`: Training options
    - `:epochs` - Number of training epochs (default: 10)
    - `:learning_rate` - Learning rate (default: 0.0001)
    - `:optimizer` - Optimizer to use (default: :adamw)
    - `:jit_compile?` - Enable JIT compilation for speed (default: true)
    - `:log_interval` - How often to log progress (default: 50 batches)
  """
  @spec train(Axon.t(), map(), Enumerable.t(), keyword()) :: map()
  def train(model, params, train_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    learning_rate = Keyword.get(opts, :learning_rate, 0.0001)
    optimizer = Keyword.get(opts, :optimizer, :adamw)
    jit_compile = Keyword.get(opts, :jit_compile?, true)
    log_interval = Keyword.get(opts, :log_interval, 50)

    # Preprocess training data
    processed_data =
      train_data
      |> Stream.map(fn batch ->
        prepare_batch(batch)
      end)

    # Cross-entropy loss for language modeling
    loss_fn = fn y_true, y_pred ->
      # y_pred shape: {batch, seq_len, vocab_size}
      # y_true shape: {batch, seq_len} (token IDs)

      # Reshape for loss computation
      {batch_size, seq_len, vocab_size} = Nx.shape(y_pred)
      y_pred_flat = Nx.reshape(y_pred, {batch_size * seq_len, vocab_size})
      y_true_flat = Nx.reshape(y_true, {batch_size * seq_len})

      # Sparse categorical cross-entropy
      Axon.Losses.categorical_cross_entropy(
        Nx.new_axis(y_true_flat, -1),
        y_pred_flat,
        reduction: :mean,
        from_logits: true,
        sparse: true
      )
    end

    # Create optimizer
    optimizer_fn =
      case optimizer do
        :adamw -> Polaris.Optimizers.adamw(learning_rate: learning_rate)
        :adam -> Polaris.Optimizers.adam(learning_rate: learning_rate)
        :sgd -> Polaris.Optimizers.sgd(learning_rate: learning_rate)
        _ -> Polaris.Optimizers.adamw(learning_rate: learning_rate)
      end

    # Train the model with optimized settings
    loop_opts = [
      epochs: epochs,
      compiler: EXLA,
      # Log progress periodically
      iterations: log_interval
    ]

    # Add JIT compilation options for better performance
    loop_opts =
      if jit_compile do
        # JIT compilation can speed up execution significantly
        Keyword.put(loop_opts, :jit_compile?, true)
      else
        loop_opts
      end

    IO.puts("\n🚀 Training Configuration:")
    IO.puts("  - Epochs: #{epochs}")
    IO.puts("  - Learning rate: #{learning_rate}")
    IO.puts("  - Optimizer: #{optimizer}")
    IO.puts("  - JIT compile: #{jit_compile}")
    IO.puts("  - Backend: EXLA (supports GPU acceleration)")
    IO.puts("\n💡 Tip: If you have a GPU, EXLA will automatically use it!")
    IO.puts("   On Apple Silicon (M1/M2/M3), Metal acceleration is automatic.\n")

    model
    |> Axon.Loop.trainer(loss_fn, optimizer_fn)
    |> Axon.Loop.metric(fn y_true, y_pred -> token_accuracy(y_true, y_pred) end, "Accuracy")
    |> Axon.Loop.handle_event(:iteration_completed, &log_metrics/1, every: log_interval)
    |> Axon.Loop.run(processed_data, params, loop_opts)
  end

  # Log training metrics for better visibility
  defp log_metrics(%{metrics: metrics, iteration: iteration} = state) do
    loss = metrics["loss"] |> Nx.to_number() |> Float.round(4)
    accuracy = Map.get(metrics, "Accuracy", Nx.tensor(0)) |> Nx.to_number() |> Float.round(4)

    IO.write("\r  Batch #{iteration}: Loss = #{loss}, Accuracy = #{accuracy * 100}%")

    {:continue, state}
  end

  # Token-level accuracy metric
  defp token_accuracy(y_true, y_pred) do
    # y_pred shape: {batch, seq_len, vocab_size}
    # y_true shape: {batch, seq_len}
    predictions = Nx.argmax(y_pred, axis: -1)
    Nx.mean(Nx.equal(predictions, y_true))
  end

  # Prepare batch for training
  defp prepare_batch(%{input_ids: input_ids, labels: labels}) do
    {batch_size, seq_len} = Nx.shape(input_ids)

    # Create position IDs (0, 1, 2, ..., seq_len-1)
    position_ids =
      Nx.iota({seq_len})
      |> Nx.tile([batch_size, 1])

    input = %{
      "input_ids" => input_ids,
      "position_ids" => position_ids
    }

    {input, labels}
  end

  @doc """
  Generates text given a prompt using the trained model.

  ## Parameters

  - `model`: The trained transformer model
  - `params`: Trained model parameters
  - `prompt_tokens`: Starting tokens (tensor of shape {1, prompt_len})
  - `opts`: Generation options
    - `:max_length` - Maximum sequence length to generate (default: 100)
    - `:temperature` - Sampling temperature (default: 1.0, higher = more random)
    - `:top_k` - Top-k sampling (default: 50)
    - `:top_p` - Nucleus sampling threshold (default: 0.9)
    - `:repetition_penalty` - Penalty for repeating tokens (default: 1.2, higher = less repetition)
    - `:no_repeat_ngram_size` - Prevent repeating n-grams (default: 3)
  """
  @spec generate(Axon.t(), map(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def generate(model, params, prompt_tokens, opts \\ []) do
    max_length = Keyword.get(opts, :max_length, 100)
    temperature = Keyword.get(opts, :temperature, 1.0)
    top_k = Keyword.get(opts, :top_k, 50)
    top_p = Keyword.get(opts, :top_p, 0.9)
    repetition_penalty = Keyword.get(opts, :repetition_penalty, 1.2)
    no_repeat_ngram = Keyword.get(opts, :no_repeat_ngram_size, 3)

    {_batch_size, prompt_len} = Nx.shape(prompt_tokens)

    # Autoregressive generation with repetition tracking
    Enum.reduce(prompt_len..(max_length - 1), prompt_tokens, fn _step, current_tokens ->
      # Predict next token with anti-repetition measures
      next_token =
        predict_next_token(
          model,
          params,
          current_tokens,
          temperature,
          top_k,
          top_p,
          repetition_penalty,
          no_repeat_ngram
        )

      # Append to sequence
      Nx.concatenate([current_tokens, Nx.reshape(next_token, {1, 1})], axis: 1)
    end)
  end

  # Predict next token given current sequence
  defp predict_next_token(
         model,
         params,
         current_tokens,
         temperature,
         top_k,
         top_p,
         repetition_penalty,
         no_repeat_ngram
       ) do
    {batch_size, seq_len} = Nx.shape(current_tokens)

    # Create position IDs
    position_ids =
      Nx.iota({seq_len})
      |> Nx.tile([batch_size, 1])

    # Get model predictions
    logits =
      Axon.predict(
        model,
        params,
        %{
          "input_ids" => current_tokens,
          "position_ids" => position_ids
        },
        compiler: EXLA
      )

    # Get logits for last position: {batch_size, vocab_size}
    last_logits = logits[[0..-1//1, -1, 0..-1//1]]

    # Apply repetition penalty - penalize tokens already generated
    last_logits = apply_repetition_penalty(last_logits, current_tokens, repetition_penalty)

    # Apply n-gram blocking - prevent repeating sequences
    last_logits = apply_ngram_blocking(last_logits, current_tokens, no_repeat_ngram)

    # Apply temperature
    scaled_logits = Nx.divide(last_logits, temperature)

    # Sample from top-k and top-p (nucleus sampling)
    sample_token(scaled_logits, top_k, top_p)
  end

  # Apply repetition penalty to discourage repeating tokens
  defp apply_repetition_penalty(logits, current_tokens, penalty) do
    if penalty == 1.0 do
      logits
    else
      # Get unique tokens in current sequence
      tokens_list = current_tokens |> Nx.to_flat_list()

      # For each token that appears, divide its logit by penalty
      Enum.reduce(tokens_list, logits, fn token_id, acc_logits ->
        # Get current value
        current_val = acc_logits[0][token_id]

        # Apply penalty (divide if positive, multiply if negative)
        penalized_val =
          if Nx.to_number(current_val) > 0 do
            Nx.divide(current_val, penalty)
          else
            Nx.multiply(current_val, penalty)
          end

        # Update the logit
        Nx.put_slice(acc_logits, [0, token_id], Nx.reshape(penalized_val, {1, 1}))
      end)
    end
  end

  # Block n-grams that would create repetition
  defp apply_ngram_blocking(logits, current_tokens, ngram_size) do
    if ngram_size < 2 do
      logits
    else
      {_batch, seq_len} = Nx.shape(current_tokens)

      # Only check if we have enough tokens
      if seq_len >= ngram_size - 1 do
        # Get last (ngram_size - 1) tokens
        context = current_tokens[[0, (seq_len - ngram_size + 1)..-1//1]]
        context_list = Nx.to_flat_list(context)

        # Find all ngrams in the sequence that start with this context
        tokens_list = Nx.to_flat_list(current_tokens)
        blocked_tokens = find_ngram_continuations(tokens_list, context_list, ngram_size)

        # Set blocked tokens to very negative value
        Enum.reduce(blocked_tokens, logits, fn token_id, acc_logits ->
          Nx.put_slice(acc_logits, [0, token_id], Nx.tensor([[-1.0e10]]))
        end)
      else
        logits
      end
    end
  end

  # Find tokens that would complete a repeated n-gram
  defp find_ngram_continuations(tokens_list, context_list, ngram_size) do
    ngram_len = ngram_size - 1

    # Slide through the sequence looking for matching contexts
    tokens_list
    |> Enum.chunk_every(ngram_size, 1, :discard)
    |> Enum.filter(fn ngram ->
      Enum.take(ngram, ngram_len) == context_list
    end)
    |> Enum.map(fn ngram -> List.last(ngram) end)
    |> Enum.uniq()
  end

  # Sample token using top-k, top-p, and proper probabilistic sampling
  defp sample_token(logits, top_k, _top_p) do
    # logits shape: {batch_size, vocab_size}
    {_batch_size, vocab_size} = Nx.shape(logits)

    # Step 1: Apply top-k filtering
    {top_k_values, top_k_indices} = Nx.top_k(logits, k: min(top_k, vocab_size))

    # Step 2: Apply softmax to get probabilities
    probs = Nx.exp(top_k_values)
    probs = Nx.divide(probs, Nx.sum(probs, axes: [-1], keep_axes: true))

    # Step 3: Apply top-p (nucleus) filtering
    # Note: Full nucleus sampling would filter based on cumulative probability
    # For now using simpler top-k approach, but keeping parameter for future enhancement

    # Step 4: Sample from the distribution (using categorical sampling)
    # Convert to flat list and sample
    probs_list = probs |> Nx.to_flat_list()
    indices_list = top_k_indices |> Nx.to_flat_list()

    # Sample token (weighted by probability)
    sampled_token = weighted_random_sample(probs_list, indices_list)

    Nx.tensor([sampled_token])
  end

  # Weighted random sampling
  defp weighted_random_sample(probs, indices) do
    # Generate random number [0, 1)
    rand = :rand.uniform()

    # Find which bucket it falls into
    {_cumsum, idx} =
      Enum.reduce_while(Enum.zip(probs, indices), {0.0, hd(indices)}, fn {prob, token_idx},
                                                                         {cumsum, _} ->
        new_cumsum = cumsum + prob

        if rand < new_cumsum do
          {:halt, {new_cumsum, token_idx}}
        else
          {:cont, {new_cumsum, token_idx}}
        end
      end)

    idx
  end

  @doc """
  Evaluates the model on test data.

  ## Parameters

  - `model`: The trained transformer model
  - `params`: Trained model parameters
  - `test_data`: Test dataset
  - `opts`: Evaluation options
  """
  @spec evaluate(Axon.t(), map(), Enumerable.t(), keyword()) :: map()
  def evaluate(model, params, test_data, _opts \\ []) do
    # Preprocess test data
    processed_data =
      test_data
      |> Stream.map(fn batch ->
        prepare_batch(batch)
      end)

    # Evaluate
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(fn y_true, y_pred -> token_accuracy(y_true, y_pred) end, "Accuracy")
    |> Axon.Loop.run(processed_data, params, compiler: EXLA)
  end

  @doc """
  Prepares tokenized text data for training.

  Converts a list of token lists into batched tensors suitable for training.

  ## Parameters

  - `token_sequences`: List of token ID sequences
  - `opts`: Preparation options
    - `:batch_size` - Batch size (default: 32, increase for better GPU utilization)
    - `:seq_len` - Sequence length (default: 128)
    - `:shuffle` - Whether to shuffle data (default: true)
    - `:max_concurrency` - Max parallel tasks for sequence creation (default: System.schedulers_online())

  ## Performance Tips

  - **Larger batch_size** (64, 128, 256) = faster training with GPU, but needs more memory
  - **Parallel processing** already enabled via Task.async_stream
  - Monitor memory usage and adjust batch_size accordingly
  """
  @spec prepare_training_data(list(list(integer())), keyword()) :: Enumerable.t()
  def prepare_training_data(token_sequences, opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 32)
    seq_len = Keyword.get(opts, :seq_len, 128)
    shuffle = Keyword.get(opts, :shuffle, true)
    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())

    IO.puts("\n📊 Data Preparation Settings:")
    IO.puts("  - Batch size: #{batch_size} (increase for faster GPU training)")
    IO.puts("  - Sequence length: #{seq_len}")
    IO.puts("  - Parallel workers: #{max_concurrency} CPU cores")
    IO.puts("  - Shuffle: #{shuffle}\n")

    # Create overlapping sequences of length seq_len + 1
    # (seq_len for input, +1 for target)
    # Using Task.async_stream for parallel processing across CPU cores
    sequences =
      token_sequences
      |> Task.async_stream(
        fn tokens ->
          create_sequences(tokens, seq_len + 1)
        end,
        timeout: :infinity,
        max_concurrency: max_concurrency,
        # Don't preserve order for speed (we shuffle anyway)
        ordered: false
      )
      |> Enum.flat_map(fn {:ok, result} -> result end)

    sequences =
      if shuffle do
        Enum.shuffle(sequences)
      else
        sequences
      end

    IO.puts(
      "Total sequences created: #{length(sequences)}, will now create #{div(length(sequences), batch_size)} batches."
    )

    # Batch sequences
    sequences
    |> Stream.chunk_every(batch_size, batch_size, :discard)
    |> Stream.map(fn batch ->
      # Convert to tensors
      input_ids =
        batch
        |> Enum.map(fn seq -> Enum.slice(seq, 0, seq_len) end)
        |> Nx.tensor()

      labels =
        batch
        |> Enum.map(fn seq -> Enum.slice(seq, 1, seq_len) end)
        |> Nx.tensor()

      %{input_ids: input_ids, labels: labels}
    end)
  end

  # Create sequences of fixed length from a longer sequence
  defp create_sequences(tokens, seq_len) do
    token_len = length(tokens)

    cond do
      # Long enough: create sliding windows
      token_len > seq_len ->
        0..(token_len - seq_len)
        |> Enum.map(fn start_idx ->
          Enum.slice(tokens, start_idx, seq_len)
        end)

      # Exact match or reasonably close: use or pad
      token_len >= max(8, div(seq_len, 4)) ->
        if token_len >= seq_len do
          [Enum.take(tokens, seq_len)]
        else
          # Pad with zeros (padding token ID 0)
          padded = tokens ++ List.duplicate(0, seq_len - token_len)
          [Enum.take(padded, seq_len)]
        end

      # Too short: skip
      true ->
        []
    end
  end
end
