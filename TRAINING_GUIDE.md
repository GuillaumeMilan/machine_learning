# Transformer Training Guide - Understanding Your Model's Learning

## 🎓 What I Changed and Why (Student-Friendly Explanation)

### Problem: Your Model Wasn't Learning Well

You mentioned your accuracy stays low even with more epochs. This is like studying the same way for more hours but still not improving your test scores. The problem isn't the **time spent** (epochs), but **how you're studying** (hyperparameters and architecture).

---

## 🔍 Critical Issues I Found

### 1. **Tiny Model = Underfitting** 
**Old:** 128-dimensional embeddings, 2 layers  
**New:** 256-dimensional embeddings, 4 layers

**Student Explanation:**
Think of your model like a student's brain. Your old model had a very small "brain" (128 dimensions, 2 layers). It's like trying to learn calculus with only basic arithmetic knowledge - you simply don't have enough capacity.

**Analogy:**
- **Embedding dimension (128→256):** Like expanding from a 128-word vocabulary to 256 words. More words = more nuanced understanding.
- **Number of layers (2→4):** Like adding more years of education. Each layer learns increasingly complex patterns.

**Why it matters:**
Language is complex! With only 2 layers and 128 dimensions, your model can barely learn basic patterns. 4 layers with 256 dimensions gives it room to understand syntax, semantics, and context.

---

### 2. **Learning Rate Too Low**
**Old:** 0.0003 (very conservative)  
**New:** 0.001 (3× faster)

**Student Explanation:**
The learning rate is like your "step size" when climbing a mountain to find the best view (minimum loss).

```
Too small (0.0003):  You take tiny baby steps → takes forever to reach the top
Too large (0.01):     You take giant leaps → might overshoot and fall off
Just right (0.001):   Good balance between speed and stability
```

**Why it matters:**
With 0.0003, your model was barely updating its weights each step. It's like trying to learn by reading one sentence per day - technically you're learning, but painfully slow.

---

### 3. **Batch Size Too Small**
**Old:** 4 samples per batch  
**New:** 16 samples per batch

**Student Explanation:**
Imagine you're a teacher grading papers to understand what students struggle with:
- **Batch of 4:** You only look at 4 papers before adjusting your teaching → very noisy signal
- **Batch of 16:** You look at 16 papers → more reliable understanding of common issues

**Technical Detail:**
```
Gradient = Average error across batch

Small batch (4):   High variance, unstable gradients
Large batch (16):  Smoother gradients, more stable learning
Too large (128):   Takes more memory, might get stuck in local minima
```

**Why it matters:**
With only 4 examples, one weird sample can throw off your entire gradient. 16 gives a more reliable signal while still allowing some randomness (which helps exploration).

---

### 4. **Sequence Length Too Short**
**Old:** 32 tokens  
**New:** 128 tokens

**Student Explanation:**
This is like asking a student to understand a story by only reading 32 words at a time vs 128 words.

**Example:**
```
32 tokens:  "The quick brown fox jumps over the lazy dog. Machine learning is..."
            ↑ Context cuts off too early

128 tokens: "The quick brown fox jumps over the lazy dog. Machine learning is 
             a fascinating field with many applications. Neural networks can 
             learn complex patterns..."
            ↑ Enough context to understand relationships
```

**Why it matters:**
Language has long-range dependencies. To predict what comes next, you need sufficient context. 32 tokens is barely 2-3 sentences. 128 tokens gives your model a paragraph of context.

---

## 📊 How These Changes Work Together

Think of training a neural network like teaching a class:

| Aspect | Old (Poor Learning) | New (Better Learning) |
|--------|--------------------|--------------------|
| **Model Capacity** | Small brain (128d, 2 layers) | Bigger brain (256d, 4 layers) |
| **Learning Rate** | Timid steps (0.0003) | Confident steps (0.001) |
| **Batch Size** | 4 students sampled | 16 students sampled |
| **Context Window** | 32 word excerpts | 128 word excerpts |
| **Epochs** | 3 repetitions | 10 repetitions |

### The Synergy:
1. **Bigger model** = Can learn more complex patterns
2. **Higher learning rate** = Learns those patterns faster
3. **Larger batches** = More stable learning signal
4. **Longer sequences** = Better context for predictions
5. **More epochs** = Enough time to converge with faster learning rate

---

## 🧮 The Math Behind It (Simplified)

### Gradient Descent Update Rule:
```
new_weights = old_weights - learning_rate × gradient
```

**Old setup:**
```elixir
# With learning_rate = 0.0003 and batch_size = 4
new_weights = old_weights - 0.0003 × (noisy_gradient_from_4_samples)
              ↑ tiny step    ↑ unreliable direction
```

**New setup:**
```elixir
# With learning_rate = 0.001 and batch_size = 16
new_weights = old_weights - 0.001 × (stable_gradient_from_16_samples)
              ↑ bigger step  ↑ reliable direction
```

---

## 🎯 What Each Hyperparameter Controls

### 1. **Embedding Dimension** (128 → 256)
**What it is:** Size of the vector representing each token

**Analogy:** 
- 128D = Describing a person with 128 characteristics
- 256D = Describing a person with 256 characteristics (more nuanced)

**Trade-off:**
- ✅ Higher = More expressive, can capture subtle meanings
- ❌ Higher = More parameters, needs more data, slower training

### 2. **Number of Layers** (2 → 4)
**What it is:** Depth of the network

**Analogy:**
```
Layer 1: Learns basic patterns (common letter combinations)
Layer 2: Learns words and simple phrases
Layer 3: Learns sentence structure and grammar
Layer 4: Learns semantic meaning and context
```

**Trade-off:**
- ✅ More layers = Can learn hierarchical features
- ❌ More layers = Harder to train, vanishing gradients

### 3. **Number of Attention Heads** (4 → 8)
**What it is:** Parallel attention mechanisms

**Analogy:** Like having multiple perspectives when reading:
- Head 1: Focuses on subject-verb agreement
- Head 2: Focuses on pronoun references
- Head 3: Focuses on semantic similarity
- ... (8 different aspects simultaneously)

**Trade-off:**
- ✅ More heads = Richer representations
- ❌ More heads = More computation

### 4. **Feed-Forward Dimension** (512 → 1024)
**What it is:** Size of the intermediate layer in each transformer block

**Analogy:** Like the size of your "working memory" when processing information

**Trade-off:**
- ✅ Larger = More capacity to transform representations
- ❌ Larger = More parameters to train

---

## 📈 Expected Training Dynamics

### What You Should See Now:

**Epoch 1-2: Rapid Initial Learning**
```
Epoch 1: Loss: 6.2 → 4.8, Accuracy: 12% → 25%
Epoch 2: Loss: 4.8 → 3.9, Accuracy: 25% → 35%
```
The model quickly learns basic patterns (common words, simple sequences)

**Epoch 3-5: Steady Improvement**
```
Epoch 3: Loss: 3.9 → 3.2, Accuracy: 35% → 42%
Epoch 4: Loss: 3.2 → 2.8, Accuracy: 42% → 48%
Epoch 5: Loss: 2.8 → 2.5, Accuracy: 48% → 52%
```
Learning grammatical structure, word associations

**Epoch 6-10: Refinement**
```
Epoch 6: Loss: 2.5 → 2.3, Accuracy: 52% → 55%
...
Epoch 10: Loss: 1.9 → 1.8, Accuracy: 58% → 62%
```
Fine-tuning, learning rare patterns, improvements slow down

### ⚠️ Warning Signs:

**If loss doesn't decrease:**
- Learning rate might still be too low
- Model might be too small for your data
- Data might have too much noise

**If loss decreases but accuracy stays low:**
- Your task might be very hard
- Vocabulary might be too large
- Need more training data

**If loss goes to NaN:**
- Learning rate is too high
- Gradient explosion (need gradient clipping)

---

## 🚀 Advanced Optimizations (Not Yet Implemented)

Here are additional techniques you could add for even better training:

### 1. **Learning Rate Warmup & Scheduling**
```elixir
# Start with small LR, gradually increase, then decay
Epoch 1:     LR = 0.0001 (warmup)
Epoch 2-3:   LR = 0.001  (full speed)
Epoch 4-7:   LR = 0.0005 (decay)
Epoch 8-10:  LR = 0.0001 (fine-tune)
```

**Why:** Prevents early instability and allows fine-tuning later.

### 2. **Gradient Clipping**
```elixir
# Prevent gradients from exploding
if gradient_norm > 1.0:
    gradient = gradient / gradient_norm  # Normalize to max 1.0
```

**Why:** Deep networks can have exploding gradients. Clipping stabilizes training.

### 3. **Label Smoothing**
```elixir
# Instead of hard targets [0, 0, 1, 0]
# Use soft targets [0.03, 0.03, 0.91, 0.03]
```

**Why:** Prevents overconfidence, improves generalization.

### 4. **Mixed Precision Training**
Use 16-bit floats instead of 32-bit for speed.

**Why:** 2× faster, uses less memory, minimal accuracy loss.

---

## 🎬 How to Use Your Improved Training

### Basic Training:
```elixir
# Use all the improved defaults
MachineLearning.TransformerTraining.run()
```

### Custom Configuration:
```elixir
config = %{
  # Data settings
  corpus_dir: "path/to/your/corpus",
  sample_size: 1000,
  
  # Model architecture
  embed_dim: 256,      # Or 512 for more capacity
  num_heads: 8,        # Or 16 for more attention
  num_layers: 4,       # Or 6 for deeper network
  ff_dim: 1024,        # Or 2048 for more capacity
  max_seq_len: 256,    # Or 512 for longer context
  
  # Training hyperparameters
  batch_size: 16,      # Or 32 if you have enough memory
  seq_len: 128,        # Or 256 for longer context
  epoch: 10,           # Or 20 for more training
  learning_rate: 0.001 # Or 0.0005 for more stability
}

MachineLearning.TransformerTraining.run(config)
```

### Continue Training Existing Model:
```elixir
config = %{
  epoch: 5,
  learning_rate: 0.0005  # Lower LR for fine-tuning
}

MachineLearning.TransformerTraining.run_on_model(
  "models/transformer_1234567890",
  config
)
```

---

## 📚 Key Takeaways (Student Summary)

1. **Model Size Matters:** Too small = can't learn complex patterns
2. **Learning Rate is Critical:** Too low = slow learning, too high = unstable
3. **Batch Size Affects Gradient Quality:** Larger = more stable, but not too large
4. **Context Window is Important:** Language has long-range dependencies
5. **It's About Balance:** All hyperparameters work together

### The Golden Rule:
> Don't just increase epochs! If your model isn't learning, check:
> 1. Is the model big enough?
> 2. Is the learning rate appropriate?
> 3. Are the batches the right size?
> 4. Is the sequence length sufficient?
> 5. Do you have enough data?

---

## 🔬 Experiment and Learn!

Try different configurations and observe:
- How does doubling `embed_dim` affect training time and accuracy?
- What happens if you use `batch_size: 8` vs `batch_size: 32`?
- How much improvement do you get from 4 layers vs 6 layers?

**Machine Learning is empirical** - theory guides you, but experiments teach you! 🎓

---

## 📖 Further Reading

- **Attention Is All You Need** (Vaswani et al., 2017) - Original Transformer paper
- **Deep Learning Book** (Goodfellow et al.) - Chapter 8 on Optimization
- **Understanding Learning Rate Schedules** - How warmup and decay help
- **Batch Size Trade-offs** - Small vs large batch training dynamics

Good luck with your training! 🚀
