# LLM Fundamentals - Complete Guide

---

## 1️⃣ What is an LLM

### Core Definition
- A neural network trained to predict the next token based on previous tokens
- "An LLM is a neural network trained to predict the next token in a sequence; tokens are the atomic units of text the model processes"

### What an LLM is NOT
- ❌ Does not "understand" like humans
- ❌ Does not store memory
- ❌ Does not know facts inherently
- ❌ Does not reason like logic engines

### How LLMs Work
```
Input tokens → Neural network → Probability distribution → Pick next token → Repeat
```

---

## 2️⃣ Tokens (not words)

### Key Distinction
- Tokens are NOT words
- Tokens are pieces of text the model actually processes
- One word ≠ one token; One token ≠ one word

### Why tokens exist
- LLMs cannot process raw characters efficiently
- LLMs cannot process entire words reliably
- Solution: Text is converted into tokens using a tokenizer

### Tokens control three key factors

| Factor | Impact |
|--------|--------|
| **Cost** (cloud LLMs) | More tokens = more money |
| **Context window** | Models have maximum token limit (e.g., 8,000 tokens) |
| **Speed** | More tokens = slower response |

**Context window includes:**
- System messages
- User messages  
- Assistant replies

---

## 3️⃣ Input tokens vs Output tokens

### Example breakdown

**When you send:** "Explain FastAPI in detail"

**Tokens used:**
- Input tokens (your prompt) ≈ 3 tokens
- Output tokens (model's answer) ≈ 50 tokens
- **Total ≈ 53 tokens**

### Tokenization example (conceptual)
```
Text:    ChatGPT is helpful
Becomes: [1345, 2987, 203, 9876]

The model never sees words, only numbers.
```

---

## 4️⃣ Why LLMs feel "smart"

### Reasons
- They've seen trillions of token patterns
- They predict the most likely continuation
- They generate fluent language

### Reality check
- It's next-token prediction, nothing more

---

## 5️⃣ Stateless nature

- ❌ LLMs do not store memory
- ❌ Chat APIs do not keep history
- ❌ Every request is independent
- **Key principle:** "LLMs are stateless; conversational memory is simulated by resending context"

---

## 6️⃣ Chat vs Generate

### Why chat "feels" stateful
*Section note: Who stores memory*

---

## 7️⃣ Message roles

**Types:** system / user / assistant

### Why system messages matter
*Section continuation in full document*

---

## 8️⃣ Student analogy

### Brain-size mapping
- **Parameters** → brain size (e.g., llama3.2b = 3.2 billion parameters)
  - "3B parameters means the model has about 3 billion learned numerical weights that operate on tokenized inputs; they are not the tokens themselves"
- **Training tokens** → books read
- **Context window** → short-term memory
- **Tokens** → words in a conversation

**Reality:** A student can read millions of books but still has one brain

---

## 9️⃣ Quantization

### Definition
Quantization is the process of reducing the precision of a model's parameters to make LLMs smaller, faster, and cheaper to run.

### Key points
- LLMs have billions of parameters (numbers)
- Normally stored as 32-bit floats (FP32) → very large memory
- Quantization stores them using fewer bits (16, 8, or 4)

### Why it's done
- ✅ Reduces RAM / VRAM usage
- ✅ Enables local inference (laptops, CPUs)
- ✅ Faster loading and execution
- ⚠️ Slight quality loss (usually minor)

### Memory comparison

| Type | Bits per parameter | Approx memory |
|------|-------------------|---------------|
| FP32 | 32 | ~12.8 GB |
| FP16 | 16 | ~6.4 GB |
| INT8 | 8 | ~3.2 GB |
| Q4 | 4 | ~1.6 GB |

*This is why Ollama models are often 2–4 GB, not 13 GB.*

### What quantization affects & doesn't affect

**Affects:**
- ✔ Memory
- ✔ Speed
- ✔ Deployment feasibility

**Does NOT affect:**
- ❌ Context window
- ❌ Training data
- ❌ Model architecture

### One-line takeaway (memorize this)
Quantization compresses model weights by reducing numerical precision, allowing large LLMs to run efficiently with minimal quality loss.

---

## 🔟 Context window

### Definition
Context window is the maximum number of tokens an LLM can see at one time in a single request.

### Key points
- Measured in tokens, not words
- Includes: System messages, User messages, Assistant replies
- Prompt + response together must fit inside it

### What happens if exceeded
- Oldest tokens are dropped (truncation), OR
- Conversation is summarized, OR
- Request fails (depends on system)

### Why it exists
- LLMs use attention, which is computationally expensive
- More tokens = more memory + slower inference

### Important clarifications
- ❌ Tokens ≠ context window
- ✔ Context window = capacity limit
- ✔ Tokens = content filling that limit

### Why models "forget"
- LLMs are stateless
- When tokens exceed the context window, earlier messages fall out
- System instructions can also be lost if too long

### One-line takeaway
The context window is the maximum number of tokens an LLM can attend to in a single request; exceeding it causes loss of earlier context.

---

## 1️⃣1️⃣ Decoding parameters (temperature, top_p, max_tokens, stop)

### Overview
```
Prompt → Tokenization → MODEL (3.2B learned weights) → Token probabilities 
→ Decoding parameters → Chosen token → Repeat
```

### Definition
Decoding parameters are runtime controls that decide how an LLM selects the next token, without changing the model's learned knowledge.

- **Model parameters** = what the model knows
- **Decoding parameters** = how it chooses words

### What they are
- Applied after the model computes token probabilities
- Do not modify model weights
- Affect style, randomness, and length, not intelligence

### Full parameter reference

| Parameter | What it controls |
|-----------|-----------------|
| **temperature** | Randomness of output |
| **top_p** | Limits choices to most probable tokens |
| **max_tokens** | Maximum tokens in response |
| **stop** | Tokens that end generation |
| **presence_penalty** | Encourages new topics |
| **frequency_penalty** | Reduces repetition |
| **seed** | Makes output reproducible |

---

## 1️⃣2️⃣ Temperature — Randomness of output

### What it actually does
- Controls how sharp or flat the probability distribution is
- Applied before sampling

### Intuition
- Low temperature → model picks the most likely token
- High temperature → model explores less likely tokens

### Temperature ranges

| Temperature | Behavior |
|-------------|----------|
| 0.0–0.2 | Deterministic, factual |
| 0.3–0.7 | Balanced |
| 0.8–1.2 | Creative |
| >1.5 | Chaotic / nonsense |

**Note:** temperature = 0 ≈ greedy decoding (no randomness) and does not add new knowledge

---

## 1️⃣3️⃣ top_p (nucleus sampling) — Limits token choices

### Core concept
The nucleus is the smallest set of next-token candidates whose combined probability mass reaches top_p.

**At a given step:**
- Model predicts probabilities for the next token only (not future tokens)
- Tokens are sorted by probability
- Added one by one
- Stop when cumulative probability ≥ top_p
- That resulting set = nucleus

### What it actually does
- Chooses the smallest set of tokens whose cumulative probability ≥ top_p
- Sampling happens only inside this set

### Intuition
- Prevents low-probability garbage tokens
- Keeps responses coherent

### Example
If probabilities are: A: 0.50, B: 0.25, C: 0.15, D: 0.05, E: 0.05
- top_p = 0.9 → {A, B, C}
- top_p = 0.6 → {A, B}

### top_p ranges

| top_p | Behavior |
|-------|----------|
| 0.9 | Safe default |
| 0.7 | Conservative |
| 0.5 | Very strict |

**Note:** 
- top_k limits the model to choosing only the K most probable next tokens (rarely used)
- Temperature reshapes probabilities; top_p selects from them — so temperature must come first

---

## 1️⃣4️⃣ max_tokens — Response length limit

### What it actually does
- Hard upper bound on generated tokens
- Includes only the output, not prompt

### Why it exists
- Prevents infinite generation
- Controls cost & latency

### Important clarifications
- If model finishes early → stops naturally
- If limit is reached → output is cut off

---

## 1️⃣5️⃣ stop — When generation must end

### What it actually does
- Defines exact token sequences that force termination
- Checked after each token

### Common use cases
- End at "User:"
- End at "###"

### Stop before leaking system prompts

**Example:**
```json
"stop": ["\n\n", "User:"]
```

When model outputs any of these → generation halts

### Important notes
- Stop tokens are not included in output
- Multiple stop sequences allowed

---

## 1️⃣6️⃣ Decoding pipeline

```
Model computes probabilities
  ↓
Temperature applied
  ↓
top_p filtering
  ↓
Token sampled
  ↓
Check stop condition
  ↓
Repeat until max_tokens or stop
```

### Common mistakes
- Using high temperature + high top_p → rambling
- Very low max_tokens → incomplete answers
- Forgetting stop → unwanted continuation

### Quick reference guide

| Use case | temperature | top_p | max_tokens |
|----------|-------------|-------|-----------|
| Factual QA | 0.2 | 0.9 | 200 |
| Chatbot | 0.6 | 0.9 | 300 |
| Creative | 0.9 | 0.95 | 500 |
| Code | 0.2 | 0.8 | 400 |

---

## 👉 Phase 1 Checkpoint

**After Phase 1, nothing about LLM APIs will feel magical.**

---

# PHASE 2: Ollama (Local LLMs)

## Goal: Comfort with local models

---

## 1️⃣7️⃣ Ollama architecture

### Definition
Ollama is a local HTTP server that loads LLM model files and exposes them via simple REST APIs.

### Ollama = Local OpenAI server on your machine
- Instead of api.openai.com → You have localhost:11434
- Instead of cloud GPUs → You use your CPU / GPU

### What happens when you run: `ollama run llama3.2`

**Internally:**
1. Ollama checks if model exists
2. If not: Downloads compressed model (zstd) → Decompresses it
3. Loads model into memory
4. Starts inference loop
5. Accepts prompts

### Important distinctions
- ❌ Ollama ≠ model
- ✔ Ollama = model runner + API server
- ✔ Model = .gguf file loaded by Ollama

---

## 1️⃣8️⃣ Model files

### Where Ollama stores models
- **Linux:** ~/.ollama/models
- **Inside:** Files like llama3.2-3b.Q4_K_M.gguf

### What a .gguf file contains
A .gguf file is not just weights. It includes:
- Model architecture (layers, heads)
- Learned weights (quantized)
- Tokenizer & vocab
- Context window size
- Metadata

**📌 Once loaded, Ollama does not need the internet.**

### zstd compression
- A compression format like zip
- Used when downloading models
- Compress large model → Decompress and store as gguf

**Summary:** Ollama downloads compressed models (zstd), stores them as quantized .gguf files, and loads them into memory for inference.

---

## 1️⃣9️⃣ CPU vs GPU inference

### Inference definition
Using already-trained weights to predict the next token

### CPU inference (default)
**How it works:**
- Uses highly optimized C++ (llama.cpp)
- Uses: SIMD, AVX / AVX2 / AVX512
- Runs on normal CPU cores

### GPU Inference
**How it works:**
- Moves heavy matrix multiplications to GPU
- Uses: CUDA (NVIDIA), Metal (Mac), ROCm (limited)

### Ollama's approach
- Auto-detects GPU
- Automatically offloads layers
- Falls back to CPU if needed
- You don't manually choose CPU/GPU in most cases

---

## 2️⃣0️⃣ Ollama APIs

### Four main API endpoints

| Endpoint | Purpose |
|----------|---------|
| **/api/chat** | Multi-turn conversation, Role-based messages |
| **/api/generate** | Single prompt → single response |
| **/api/embeddings** | text → vectors |
| **/api/tags** | List installed models along with metadata |

*Refer to llm-basics.md for detailed examples*

---

## 2️⃣1️⃣ Streaming responses

### Definition
The model sends the response token by token (or chunk by chunk) instead of waiting for the full answer.

### Without streaming
- Tokens are generated
- Buffered
- Sent only after completion

### With streaming
- Tokens are sent as soon as they are generated

### Why it matters for UX
- Instant feedback
- Feels fast & alive
- Essential for chat apps

*Check in llm_basics.py lines 19-40 for implementation examples*

---

# PHASE 3: OpenAI-style APIs (Cloud LLMs)

## Goal: Switch providers without confusion

---

## 2️⃣2️⃣ OpenAI ChatCompletion-style APIs

### Key principle
Same concepts, different URLs. Only:
- URLs change
- Auth is added
- Limits exist

**📌 Ollama APIs and OpenAI APIs are conceptually the same.**

---

## 2️⃣3️⃣ Rate limits (new constraint)

### Definition
Cloud models are shared. Limits exist on:
- Requests per minute (RPM)
- Tokens per minute (TPM)
- If exceeded → errors

| Limit | Meaning |
|-------|---------|
| RPM | Requests per minute |
| TPM | Tokens per minute |

### Example scenario

**Your limits:** 60 RPM, 90,000 TPM

**You send:**
- 10 requests
- Each = 10,000 tokens (input + output)
- ❌ 100,000 tokens → 429 error
- Even though RPM is OK

---

## 2️⃣4️⃣ API keys

### Why API keys exist
- Identify user
- Enforce billing
- Apply rate limits

### How they are used
```
Authorization: Bearer sk-xxxx
```

### Summary
Cloud LLM APIs do not change how LLMs work — they only add authentication, billing, and limits.

---

## 2️⃣5️⃣ Error handling & retries

### Timeouts
"The server didn't respond within the time I'm willing to wait"

| Type | Where |
|------|-------|
| Connection timeout | Network / DNS |
| Read timeout | Model is slow |
| Client timeout | Your SDK limit |

---

## 2️⃣6️⃣ 429 errors (Rate Limit)

### What triggers 429?

**Common causes:**
- Too many parallel requests
- Large prompts
- Streaming many tokens
- Agent loops

### Example error response
```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Too many requests"
  }
}
```

### Bad code (naive)
```python
for query in queries:
    call_llm(query)
```

### Good code (with retry + backoff)
```python
import time
import random

def call_with_retry(fn, retries=5):
    for i in range(retries):
        try:
            return fn()
        except RateLimitError:
            sleep = (2 ** i) + random.random()
            time.sleep(sleep)
    raise Exception("Max retries exceeded")
```

### Why exponential backoff works
If everyone retries immediately → thundering herd

**Progression:** 1s → 2s → 4s → 8s → success

**Allows:**
- Token bucket to refill
- Queue to clear

---

## 2️⃣7️⃣ 500-level errors (server-side failures)

| Code | Meaning |
|------|---------|
| 500 | Internal error |
| 502 | Bad gateway |
| 503 | Service unavailable |
| 504 | Gateway timeout |

### Retry implementation
```python
import time
import random

def call_llm_with_retry(fn, retries=5):
    for i in range(retries):
        try:
            return fn()
        except (TimeoutError, ServerError) as e:
            sleep = min(60, (2 ** i) + random.random())
            time.sleep(sleep)
    raise Exception("LLM unavailable")
```

### Error handling matrix

| Error | Retry? | Delay |
|-------|--------|-------|
| Timeout | ✅ | Exponential |
| 500 | ✅ | Exponential |
| 502/503 | ✅ | Longer |
| 429 | ✅ | Provider-specified |
| 400 | ❌ | Fix request |
| 401 | ❌ | Fix auth |

---

# PART A — Embeddings (Foundation)

---

## 2️⃣8️⃣ What embeddings are and token vs embeddings

### Definition
An embedding is a numerical vector that represents meaning.

- Not text
- Not tokens
- **Meaning**

### Why embeddings exist

**LLMs:**
- Generate text
- Are bad at searching large knowledge bases

**Embeddings:**
- Convert text → numbers
- Enable similarity search

### Example (conceptual)
```
"FastAPI is a Python framework"
→ [0.13, -0.44, 0.82, ..., 0.09]  (1536 dimensions)

These vectors are close in space.
```

### What vectors actually encode

**They encode:**
- Topic
- Intent
- Semantics
- Relationships

**They do NOT encode:**
- ❌ Grammar
- ❌ Exact wording
- ❌ Order (mostly)

### Token vs Embedding comparison

| Concept | Purpose |
|---------|---------|
| Tokens | Generation |
| Embeddings | Search & retrieval |

**Embeddings turn meaning into geometry.**

### Common misconceptions
- ❌ Embeddings are not model parameters
- ❌ Embeddings are not learned per query
- ❌ Embeddings are not context windows
- ✔ They are outputs of a trained embedding model

---

## 2️⃣9️⃣ Where embeddings live in RAG

### RAG pipeline
```
Docs → embeddings → vector DB
Query → embedding → similarity search
Top chunks → prompt → LLM
```

### Why embeddings make LLMs "know things"

**LLMs:**
- Don't store your documents

**RAG:**
- Retrieves relevant docs at runtime
- Injects them into prompt

**Result:** Knowledge is externalized

---

## 3️⃣0️⃣ Embedding models and Dimensionality

### What is an embedding model?
An embedding model is a neural network trained to map text → vectors such that semantic similarity = geometric closeness.

📌 **Different from a chat/generation model.**

### How embedding models are trained (conceptual)

**Trained on:**
- Sentence pairs
- Question–answer pairs
- Paraphrases
- Contrastive learning

**Training goal:**
- Similar meaning → vectors close
- Different meaning → vectors far

### Types of embedding models

**🔹 Proprietary (Cloud):**
- OpenAI text-embedding-3-large
- Cohere
- Google

**🔹 Open-source:**
- SentenceTransformers
- BGE (BAAI)
- E5
- GTE
- Instructor models

**🔹 Local (Ollama):**
- nomic-embed-text
- mxbai-embed-large
- bge-base

---

## 3️⃣1️⃣ Dimensionality

### Dimension sizes by model

| Model | Dimensions |
|-------|-----------|
| OpenAI small | 768 |
| OpenAI large | 3072 |
| BGE-base | 768 |
| BGE-large | 1024 |
| nomic | 768 |

### Speed vs Accuracy tradeoff

| Lower dim | Higher dim |
|-----------|-----------|
| Faster | More accurate |
| Less memory | Better nuance |
| Cheaper | Slower |

**📌 768–1024 is the industry sweet spot.**

### One crucial rule (people mess this up)
**Query and documents MUST use the same embedding model.**
Mixing models = broken similarity search.

---

## 3️⃣2️⃣ Embedding normalization

### What it means
Most models output vectors that are:
- Already normalized OR
- Should be normalized

### Why normalize?
- Makes cosine similarity stable
- Improves indexing
- Many vector DBs auto-normalize

---

## 3️⃣3️⃣ When embeddings FAIL

### Common failure cases
- Very short queries ("yes", "ok")
- Exact keyword search
- Numbers / IDs
- Highly structured data

**Solution:** Use hybrid search (later topic)

### Mental model
- Embedding model = semantic encoder
- Vector DB = memory
- LLM = reasoning engine

---

## 3️⃣4️⃣ Similarity metrics (cosine, dot, L2)

### Why similarity metrics exist
Once you have embeddings (vectors), you need to answer:
- "How close are these two meanings?"
- That's what similarity metrics do

### The three main similarity metrics

| Metric | Used for |
|--------|----------|
| Cosine similarity | Most common |
| Dot product | Fast, ranking |
| L2 (Euclidean) distance | Geometry-based |

---

## 3️⃣5️⃣ Cosine similarity (most important)

### What it measures
Measures the angle between two vectors, not their length.

### Why cosine is king 👑
- Ignores magnitude
- Focuses on direction (meaning)
- Stable across embedding models
- Works well with normalized vectors

📌 **Most embedding models are trained expecting cosine similarity.**

### Cosine value interpretation

| Cosine value | Meaning |
|--------------|---------|
| 1.0 | Identical meaning |
| 0.8 | Very similar |
| 0.5 | Somewhat related |
| 0.0 | Unrelated |
| -1.0 | Opposite meaning |

---

## 3️⃣6️⃣ Dot product

### What it measures
Measures both direction and magnitude

### When dot product is used
- Vectors are normalized
- Speed is critical
- Ranking is more important than exact similarity

📌 **If vectors are normalized: dot product ≈ cosine similarity**
That's why some DBs use dot product internally.

---

## 3️⃣7️⃣ L2 (Euclidean) distance

### What it measures
Straight-line distance between vectors

### Why L2 is less popular
- Sensitive to magnitude
- Worse semantic behavior
- Less aligned with training objectives

### Where L2 is used
- Vision models
- Older embedding systems

---

## 3️⃣8️⃣ Why cosine works best for text

### Text meaning characteristics
- Directional
- Relative
- Scale-independent

**Cosine captures exactly that.**

---

## 3️⃣9️⃣ Vector DB perspective

### Default metrics by database

| DB | Default |
|----|---------|
| FAISS | Inner product |
| Chroma | Cosine |
| Pinecone | Cosine |
| Weaviate | Cosine |

### Common mistake (critical)
- ❌ Using cosine on non-normalized vectors
- ❌ Mixing similarity metrics between indexing & querying

📌 **Index metric == query metric (must match).**

---

## 4️⃣0️⃣ Understanding dimensions

### What "dimension" really means
- One dimension = one learned semantic feature
- An embedding of size 768 means: 768 independent semantic signals
- Not human-interpretable, but statistically meaningful

### Why embeddings have FIXED size
Neural networks require:
- Fixed input size
- Fixed output size

**So:**
- 1 sentence → 768 numbers
- 1 paragraph → 768 numbers
- 1 page → 768 numbers

(for that specific model)

---

## 4️⃣1️⃣ Why not 10 dimensions? Why not 1 million?

### Too few dimensions
- Can't represent nuance
- Many meanings collapse together
- Poor retrieval quality

### Too many dimensions
- Slow search
- High memory
- Harder indexing
- Diminishing returns

### Why 768 / 1024 became standard

These numbers come from:
- Transformer hidden sizes
- Powers of 2
- Hardware efficiency

**Example:**

| Model family | Hidden size |
|--------------|------------|
| BERT-base | 768 |
| RoBERTa | 768 |
| BGE-base | 768 |
| BGE-large | 1024 |

📌 **Embedding head often mirrors hidden size.**

---

## 4️⃣2️⃣ Curse of dimensionality

### What happens as dimensions increase
- Distance between points becomes less meaningful
- Everything starts to look "far"
- Indexing gets harder

### Vector DBs combat this with
- Approximate Nearest Neighbor (ANN)
- Quantization
- Clustering

### Practical dimension tradeoffs

| Use case | Recommended dims |
|----------|-----------------|
| Small app | 384–768 |
| RAG systems | 768–1024 |
| High-precision search | 1024–1536 |
| Edge / mobile | ≤384 |

---

## 4️⃣3️⃣ Can you reduce dimensions?

### Methods to reduce
- PCA
- Autoencoders
- Quantization

### Trade-offs
- ❌ Usually hurts retrieval
- ✔️ Useful for memory-constrained systems

### Mental model
- Dimensions = semantic resolution
- More dims = sharper meaning
- Fewer dims = blurrier meaning

---

## 4️⃣4️⃣ When Embeddings FAIL (in RAG)

### Core insight
Embeddings are semantic, not factual. They capture meaning similarity, not truth or exactness.

### Major failure case 1: Very short queries

**Examples:**
- "yes"
- "ok"
- "why?"

**Problem:**
- ➡️ Too little semantic signal
- ➡️ Vectors are noisy

**Fix:** Expand query or use conversation context

### Major failure case 2: Keyword-heavy queries

**Examples:**
- Error codes (ERR_CONN_RESET)
- IDs (order_839201)
- File names

**Problem:** Embeddings blur exact tokens

**Fix:** Keyword search or hybrid search

### Major failure case 3: Numerical & tabular data

**Examples:**
- Prices
- Dates
- Metrics

**Problem:** Embeddings don't preserve numeric precision

**Fix:** Structured DB + RAG

### Major failure case 4: Domain mismatch

**Problem:** Embedding model not trained on:
- Telecom logs
- Kernel traces
- Medical codes
- ➡️ Similarity becomes meaningless

**Fix:** Domain-specific embeddings

### Major failure case 5: Long documents, bad chunking

**Problems:**
- Important info split across chunks
- Context lost

**Fix:** Smarter chunking

### Major failure case 6: False positives (semantic drift)

**Example:**
- Query: "How to reset router?"
- Retrieved: "How to restart application"
- Semantically similar but wrong

---

## 4️⃣5️⃣ Why embeddings don't "understand"

### What they optimize
- Similar meaning → close vectors

### What they do NOT optimize
- ❌ Logical correctness
- ❌ Temporal truth
- ❌ Causality
- That's LLM's job

---

## 4️⃣6️⃣ Warning sign in RAG

**If your RAG answers:**
- Confidently wrong
- With irrelevant citations
- ➡️ **Retrieval is broken, not generation**

---

## 4️⃣7️⃣ Embeddings vs search engines

| Feature | Embeddings | Keyword search |
|---------|-----------|-----------------|
| Semantic | ✅ | ❌ |
| Exact match | ❌ | ✅ |
| Numbers | ❌ | ✅ |
| Speed | Medium | Fast |

➡️ **That's why hybrid search exists.**

**Mental model:**
- Embeddings = fuzzy semantic lens
- Keyword search = sharp literal lens
- **Best systems use both**

---

# PART B — Vector Databases

---

## 4️⃣8️⃣ Why do we even need Vector Databases?

### The problem
LLMs don't "search text" — they compare meanings.

**Imagine:**
- 1 question embedding
- 1 million document embeddings
- Need to answer: "Which documents are semantically closest?"

**This is vector similarity search, not keyword search.**

### What happens without a vector DB?

```python
for each vector in database:
    compute similarity
```

**Complexity: O(N × D)**
- N = number of vectors
- D = embedding dimensions

**With N = 1,000,000, D = 768:**
- ❌ Too slow
- ❌ Too expensive
- ❌ Not scalable

### What a Vector DB gives you

A vector database:
- Stores embeddings efficiently
- Builds ANN indexes (Approximate Nearest Neighbor)
- Finds top-K closest vectors in milliseconds
- Handles metadata filtering
- Scales to millions/billions of vectors

**Key idea:** We trade perfect accuracy for massive speed gains

---

## 4️⃣9️⃣ Why SQL / Traditional DBs aren't enough

### SQL is great at

| Task | Works? |
|------|--------|
| `WHERE age > 30` | ✅ |
| `JOIN users ON orders` | ✅ |
| Exact match | ✅ |
| Range queries | ✅ |

### SQL is terrible at

| Task | Why |
|------|-----|
| Semantic similarity | No vector math |
| High-dim cosine search | No native ANN |
| Scaling similarity search | Full table scans |
| Top-K nearest neighbors | Brutal performance |

**Example:** Even with vectors stored as arrays:
```sql
SELECT * FROM docs
ORDER BY cosine_similarity(vec, query_vec)
LIMIT 5;
```
This forces full scan every time.

### Traditional DB vs Vector DB comparison

| Traditional DB | Vector DB |
|----------------|-----------|
| Structured data | Unstructured meaning |
| Exact logic | Fuzzy similarity |
| Rows & columns | Points in space |
| Deterministic | Probabilistic |

---

## 5️⃣0️⃣ What is ANN (Approximate Nearest Neighbor)?

### Exact Nearest Neighbor (ENN)
- Computes distance to every vector
- Accurate
- Unusable at scale

### ANN (what everyone uses)
- Uses clever indexing tricks
- Narrows search space
- Finds very close neighbors (not mathematically perfect)
- 100×–1000× faster

**Important truth:** In RAG, "almost correct" retrieval is MORE than enough. LLMs are robust to slight noise.

---

## 5️⃣1️⃣ ANN Indexing — Core intuition

### The question ANN answers
Imagine vectors as points in space. ANN tries to answer: "Which small region of space should I search instead of everything?"

### Common ANN ideas (high-level)

**1️⃣ Space partitioning**
- Divide vector space into regions
- Only search nearby regions

**2️⃣ Graph-based navigation**
- Each vector links to neighbors
- Traverse graph from entry point

**3️⃣ Clustering**
- Group similar vectors
- Search only top clusters

---

## 5️⃣2️⃣ Why ANN indexing is a separate step

### Pipeline
```
Text → Embedding → Normalize → Store → Index → Search
```

### Key rules
- Indexing is built after vectors are stored
- Index depends on:
  - Distance metric (cosine / L2 / dot)
  - Vector dimension
  - Dataset size

**If you:**
- Change embedding model ❌
- Change normalization ❌
- Change distance metric ❌
- ➡️ Rebuild

---

## 5️⃣3️⃣ Types of Vector Databases (preview)

| DB | Characteristic |
|----|-----------------|
| FAISS | Fastest, in-memory |
| Chroma | Dev-friendly, metadata |
| Pinecone | Managed, scalable |
| Weaviate | Schema + hybrid search |
| Milvus | Massive scale |
| PostgreSQL + pgvector | SQL + vectors |

---

## 5️⃣4️⃣ Indexing Methods (ANN Deep Dive)

### The problem to solve

You have:
- N vectors (documents)
- 1 query vector
- A distance metric (cosine / L2 / dot)

**Goal:** Find top-K closest vectors without scanning all N

**Question indexing answers:** "Which small subset of vectors should I even look at?"

---

## 5️⃣5️⃣ 1️⃣ Flat Index (Baseline — No ANN)

### What it is
- Store vectors as-is
- On search → compare against every vector
- Complexity: O(N × D)

### Pros
- ✅ Exact results
- ✅ Simple
- ✅ No preprocessing

### Cons
- ❌ Extremely slow at scale
- ❌ No pruning

### When it's used
- N < 10k
- Evaluation / testing
- Gold-standard accuracy checks

📌 **Important:** Every ANN index is compared against Flat for accuracy

---

## 5️⃣6️⃣ 2️⃣ IVF — Inverted File Index (Clustering-based)

### Core idea
"Don't search everywhere — search only relevant clusters"

### How IVF works

**Step 1: Train centroids**
- Run k-means on vectors
- Produce nlist centroids

**Step 2: Assign vectors**
- Each vector goes to nearest centroid
- Centroid A → [v1, v7, v103]
- Centroid B → [v2, v9, v55]

**Step 3: Query time**
- Embed query
- Find nearest nprobe centroids
- Search only vectors inside those centroids

### IVF parameters

| Parameter | Meaning |
|-----------|---------|
| `nlist` | Number of clusters |
| `nprobe` | How many clusters to search |

### Pros
- ✅ Massive speedup
- ✅ Good for millions of vectors
- ✅ Tunable accuracy

### Cons
- ❌ Needs training
- ❌ Bad if clusters are poor
- ❌ Recall depends on nprobe

### Where IVF shines
- Large static datasets
- Embeddings don't change often
- Disk-backed indexes

---

## 5️⃣7️⃣ 3️⃣ HNSW — Hierarchical Navigable Small World Graph

### Core idea
"Vectors form a graph; similar vectors are neighbors"

### How HNSW works

Instead of clustering, HNSW:
- Builds a multi-layer graph
- Higher layers = fewer nodes
- Lower layers = dense connections

```
Layer 3 (sparse)
   ↓
Layer 2
   ↓
Layer 1
   ↓
Layer 0 (dense, full graph)
```

### Query process
- Start at top layer
- Greedily move to closer neighbors
- Drop down layers
- Final fine search at bottom
- 🚀 No clustering. No scanning. Just graph traversal.

### HNSW parameters

| Parameter | Meaning |
|-----------|---------|
| `M` | Number of neighbors per node |
| `efConstruction` | Index build quality |
| `efSearch` | Search accuracy vs speed |

**Bigger efSearch = better recall**

### Pros
- ✅ Extremely fast
- ✅ High recall
- ✅ No training step
- ✅ Dynamic inserts supported

### Cons
- ❌ High memory usage
- ❌ Complex internals

**📌 HNSW is the default choice unless you have a strong reason not to. Most modern vector DBs use HNSW internally.**

---

## 5️⃣8️⃣ 4️⃣ PQ — Product Quantization (Compression)

### Core idea
"Store approximate vectors using fewer bytes"

### How PQ works
- Split vector into sub-vectors
- Quantize each part separately
- Store codes instead of floats

**Example:**
- 768-d float vector → ~3KB
- PQ compressed → ~64–128 bytes

### Why PQ exists
- Memory is expensive
- Disk I/O is slow
- PQ allows billions of vectors

### PQ results

| Aspect | Result |
|--------|--------|
| Memory | 🔥 Huge win |
| Speed | 🔥 Faster cache |
| Accuracy | ❌ Some loss |

### PQ combination
PQ is rarely used alone. Usually combined with IVF: **IVF+PQ**

---

## 5️⃣9️⃣ Hybrid indexes

| Combo | Why |
|-------|-----|
| Flat | Ground truth |
| IVF | Large datasets |
| HNSW | High recall + speed |
| IVF + PQ | Massive scale |
| HNSW + filters | Metadata-aware search |

---

## 6️⃣0️⃣ Distance metrics & indexing compatibility

**This is critical.**

| Metric | Notes |
|--------|-------|
| Cosine | Requires normalization |
| Dot product | Often with normalized vectors |
| L2 | Raw vectors |

---

## 6️⃣1️⃣ About FAISS

**FAISS is NOT just a vector DB.**

It is a library of ANN indexes. It provides:
- Flat
- IVF
- PQ
- IVFPQ
- HNSW

**Most vector databases internally use FAISS or FAISS-like algorithms.**

---

## 6️⃣2️⃣ Accuracy vs speed vs memory triangle

You can only optimize two:

| Optimize | Sacrifice |
|----------|-----------|
| Speed + Accuracy | Memory |
| Speed + Memory | Accuracy |
| Accuracy + Memory | Speed |

**ANN is engineering tradeoffs, not magic.**

---

## 6️⃣3️⃣ FAISS (In-memory ANN engine, performance king)

### What exactly is FAISS?

**FAISS = Facebook AI Similarity Search**

**Important correction to lock in:**
- ❌ FAISS is NOT a database
- ✔ FAISS is an ANN indexing + search library

### What it does
- Vector storage (in RAM / mmap)
- ANN indexing
- Ultra-fast similarity search

### What it does NOT do
- ❌ Handle metadata well
- ❌ Do filtering
- ❌ Handle persistence like a DB
- ❌ Provide auth / scaling / replication

**FAISS = engine, not platform.**

---

## 6️⃣4️⃣ Why FAISS exists (the real reason)

### Before FAISS
- Academic ANN code
- Inconsistent performance
- No GPU support
- Hard to scale beyond millions

### FAISS solved
- High-dimensional similarity search
- CPU + GPU acceleration
- Pluggable index types
- Production-grade speed

**Today:** Almost every vector DB is either built on FAISS or re-implements its ideas

---

## 6️⃣5️⃣ FAISS architecture

```
Embeddings (float vectors)
        ↓
FAISS Index
   ├─ Flat
   ├─ IVF
   ├─ HNSW
   ├─ PQ / IVFPQ
        ↓
Top-K IDs + distances
```

**FAISS does only one thing:**
Given a query vector → return nearest vector IDs

---

## 6️⃣6️⃣ Core FAISS index types (what actually matters)

### 1️⃣ IndexFlat (Exact)
```
IndexFlatL2
IndexFlatIP
```
- No ANN
- Exact search
- Baseline

**Use when:**
- Small dataset
- Measuring recall

### 2️⃣ IndexIVFFlat
```
IndexIVFFlat(quantizer, d, nlist)
```
- IVF clustering
- Flat vectors inside clusters

**Key params:**
- nlist → number of clusters
- nprobe → clusters searched at query

**Good for:**
- Millions of vectors
- Disk-backed indexes

### 3️⃣ IndexHNSWFlat
```
IndexHNSWFlat(d, M)
```
- Graph-based ANN
- Very fast
- High recall

**Use when:**
- Low latency matters
- RAM is available
- Dynamic inserts needed

### 4️⃣ IndexIVFPQ (Scale monster)
```
IndexIVFPQ(d, nlist, m, nbits)
```
- IVF + Product Quantization
- Massive compression

**Use when:**
- Tens / hundreds of millions
- Memory constrained

---

## 6️⃣7️⃣ FAISS + Distance metrics (CRITICAL)

### FAISS supports
- L2
- Inner Product (dot)

📌 **Cosine similarity is NOT native**

### Cosine trick
Normalize vectors → use dot product

**If you forget normalization:**
- ❌ Garbage results
- ❌ Silent failure (no error)

---

## 6️⃣8️⃣ GPU FAISS (why it's special)

### GPU FAISS capabilities
- Runs ANN search on GPU
- Blazing fast for large batches
- Used by Meta, Google-scale systems

### Trade-offs
- GPU memory limits
- Transfer overhead
- Harder to deploy

### Used when
- High QPS
- Batch queries
- Embedding pipelines on GPU

---

## 6️⃣9️⃣ FAISS lifecycle in real RAG systems

### Typical flow
1. Embed documents
2. Normalize embeddings
3. Add to FAISS index
4. Build index (train if needed)
5. Save index to disk

### At query time
```
Query → embed → normalize → FAISS.search → IDs
```

---

## 7️⃣0️⃣ Why FAISS is NOT enough alone

| Feature | FAISS |
|---------|-------|
| Metadata filtering | ❌ |
| Persistence | ⚠️ manual |
| Distributed search | ❌ |
| REST API | ❌ |
| Multi-tenant | ❌ |

**That's why people wrap FAISS inside:**
- Chroma
- Weaviate
- Milvus
- Custom services

---

## 7️⃣1️⃣ When should YOU use FAISS directly?

### Use FAISS if
- ✅ You want maximum performance
- ✅ You control the infrastructure
- ✅ You're okay writing glue code
- ✅ You don't need complex filters

### Avoid FAISS if
- ❌ You want fast prototyping
- ❌ You need metadata filters
- ❌ You want managed scaling

---

## 7️⃣2️⃣ FAISS vs Vector DBs (truth table)

| Feature | FAISS | Vector DB |
|---------|-------|-----------|
| ANN algorithms | ✅ | ✅ |
| Metadata | ❌ | ✅ |
| Persistence | ⚠️ | ✅ |
| Scaling | ❌ | ✅ |
| Ease of use | ❌ | ✅ |

**FAISS is the engine. Vector DBs are the cars built on it.**

---

## 7️⃣3️⃣ Key takeaways on FAISS
- FAISS = gold standard ANN library
- In-memory, ultra-fast
- Requires discipline (normalization, index rebuilds)
- Forms the foundation of many vector DBs
- Best for performance-critical systems

---

## 7️⃣4️⃣ Chroma DB (Developer-first vector database)

### 1️⃣ What is Chroma?

Chroma is an open-source vector database designed for:
- Local development
- Prototyping RAG systems
- Metadata-heavy workflows
- Tight integration with LLM frameworks

**Think of Chroma as:** FAISS + persistence + metadata + DX

---

## 7️⃣5️⃣ Why Chroma exists

### FAISS problems
- No metadata filtering
- No persistence by default
- No simple API
- Easy to misuse

### Chroma solves
- Simple Python API
- Automatic persistence
- Metadata storage
- Filters (where, where_document)
- Seamless LangChain / LlamaIndex usage

---

## 7️⃣6️⃣ Chroma architecture (conceptual)

```
Documents + Metadata
        ↓
Embedding Function
        ↓
Chroma Collection
   ├─ Vectors
   ├─ Metadata index
   ├─ ANN index (HNSW)
        ↓
Similarity Search + Filters
```

### Internally
- Uses HNSW
- Uses SQLite / DuckDB for metadata
- Stores vectors on disk

---

## 7️⃣7️⃣ Core concepts in Chroma (very important)

### 1️⃣ Collection
- A collection = logical namespace
- Example: `collection = chroma_client.create_collection("docs")`
- Rule: One embedding model per collection
- Mixing embeddings = broken similarity

### 2️⃣ Documents
- Raw text chunks
- Example: `documents = ["Reset the router...", "Check power cable..."]`

### 3️⃣ Metadata
- Structured filters
- Example:
```python
metadata = [
    {"source": "manual", "page": 4},
    {"source": "faq", "page": 1}
]
```

### 4️⃣ IDs
- User-defined or auto-generated
- Must be:
  - Unique
  - Stable (important for updates)

---

## 7️⃣8️⃣ How Chroma search works

### Query pipeline
```
Query text
   ↓
Embedding
   ↓
HNSW ANN search
   ↓
Metadata filter
   ↓
Top-K documents
```

**Note:** Filtering happens after ANN narrowing, not before.

---

## 7️⃣9️⃣ Why metadata filtering matters (RAG reality)

### Real RAG questions
- "Only search logs from last week"
- "Only config files"
- "Only RU alarms"

### Without metadata
- ❌ Irrelevant chunks pollute context
- ❌ LLM hallucinations increase

**Chroma makes metadata first-class, not an afterthought.**

---

## 8️⃣0️⃣ Chroma persistence model (what actually happens)

**Chroma:**
- Writes vectors + metadata to disk
- Reloads on restart
- No manual save/load needed

**Trade-off:**
- Slower than raw FAISS
- But far safer

---

## 8️⃣1️⃣ Strengths of Chroma

- ✅ Very easy to use
- ✅ Great for RAG experiments
- ✅ Metadata filtering built-in
- ✅ Open source
- ✅ Plays well with LangChain

---

## 8️⃣2️⃣ Limitations (important to know)

- ❌ Not designed for massive scale
- ❌ Single-node focus
- ❌ Limited index customization
- ❌ Not ideal for high-QPS production

**Chroma is:** Dev & prototype DB — not infra-grade

---

## 8️⃣3️⃣ Chroma vs FAISS (practical view)

| Aspect | FAISS | Chroma |
|--------|-------|--------|
| Speed | 🔥🔥🔥 | 🔥🔥 |
| Metadata | ❌ | ✅ |
| Persistence | ❌ | ✅ |
| Ease of use | ❌ | ✅ |
| Scale | Huge (manual) | Small–Medium |

---

## 8️⃣4️⃣ When should YOU use Chroma?

### Use Chroma if
- You're learning RAG
- You want quick iteration
- Dataset < few million chunks
- Metadata matters

### Avoid Chroma if
- You need distributed search
- You need strict latency SLOs
- You expect heavy concurrency

---

## 8️⃣5️⃣ Key takeaways on Chroma
- Chroma = developer-first vector DB
- Built on ANN (HNSW)
- Strong metadata support
- Perfect for learning & prototyping
- Not a FAISS replacement — a wrapper

---

## 8️⃣6️⃣ Vector DB Alternatives (Why so many exist & what problem each solves)

**All vector DBs solve similarity search, but they optimize different constraints:**
- Scale
- Cost
- Dev experience
- Filtering
- Cloud vs self-hosted

---

## 8️⃣7️⃣ 1️⃣ Pinecone — Managed & production-ready

### What Pinecone is
- Fully managed vector DB (SaaS)
- No infra to manage
- Built for production RAG

### Why Pinecone exists

**Problem:**
- FAISS & open-source DBs need ops
- Scaling ANN is hard
- High availability is painful

**Pinecone solves:**
- Auto-scaling
- Replication
- Backups
- High QPS (queries per second)
- Global availability

### Strengths
- ✅ Zero infra
- ✅ High reliability
- ✅ Metadata filtering
- ✅ Fast ANN
- ✅ Good for startups

### Weaknesses
- ❌ Cost
- ❌ Black-box internals
- ❌ Vendor lock-in

### When to use Pinecone
- Production RAG
- Customer-facing apps
- Teams without infra expertise

---

## 8️⃣8️⃣ 2️⃣ Weaviate — Schema + Hybrid Search

### What Weaviate is
- Open-source vector DB
- Strong schema support
- Built-in hybrid search

### Key differentiator: Hybrid search
- Keyword score + Vector score
- This helps when:
  - Keywords matter (IDs, error codes)
  - Semantic search alone fails

### Strengths
- ✅ Hybrid search
- ✅ Rich metadata filtering
- ✅ Graph-like schema
- ✅ Self-host or cloud

### Weaknesses
- ❌ More complex
- ❌ Higher learning curve
- ❌ Slower than pure ANN engines

### When to use Weaviate
- Mixed keyword + semantic search
- Knowledge graphs
- Enterprise schemas

---

## 8️⃣9️⃣ 3️⃣ Milvus — Massive scale (infra-grade)

### What Milvus is
- Distributed vector DB
- Designed for billions of vectors
- CNCF-backed

### Why Milvus exists

**Single-node DBs fail when:**
- Dataset grows huge
- RAM is limited
- QPS explodes

**Milvus solves:**
- Sharding
- Replication
- Disk-based ANN
- Cloud-native deployments

### Strengths
- ✅ Extreme scale
- ✅ Multiple ANN algorithms
- ✅ Cloud-native
- ✅ Used by large enterprises

### Weaknesses
- ❌ Heavy infra
- ❌ Complex ops
- ❌ Overkill for most RAGs

### When to use Milvus
- Very large corpora
- Enterprise / telecom scale
- Strict SLA systems

---

## 9️⃣0️⃣ 4️⃣ pgvector — SQL + vectors

### What pgvector is
- Vector type inside PostgreSQL
- ANN indexes (HNSW, IVF)
- SQL-first approach

### Why pgvector exists

**Problem:**
- Teams already use PostgreSQL
- Don't want new DBs

**pgvector gives:**
- Vectors in SQL
- Metadata joins
- Transactions

### Strengths
- ✅ Familiar SQL
- ✅ ACID guarantees
- ✅ Easy integration
- ✅ Good for small–medium scale

### Weaknesses
- ❌ Slower than dedicated engines
- ❌ Limited ANN tuning
- ❌ Not ideal for massive scale

### When to use pgvector
- Existing PostgreSQL infra
- Moderate vector counts
- Heavy relational metadata

---

## 9️⃣1️⃣ Comparison table (truth, not marketing)

| DB | Best at | Scale | Ops |
|-------|---------|-------|-----|
| FAISS | Raw speed | Huge | Manual |
| Chroma | Dev & RAG | Small–Med | Easy |
| Pinecone | Prod SaaS | Large | Zero |
| Weaviate | Hybrid search | Large | Medium |
| Milvus | Massive scale | Huge | Heavy |
| pgvector | SQL integration | Small–Med | Easy |

---

## 9️⃣2️⃣ Choosing wrong = bad RAG

### Common mistakes
- Pinecone for tiny experiments → 💸
- Chroma for high-QPS prod → 🔥
- Milvus for prototypes → 😵
- FAISS without metadata → 🤯

---

## 9️⃣3️⃣ Key takeaways on Vector DB Alternatives
- No "best" vector DB
- Choose based on:
  - Scale
  - Metadata needs
  - Infra maturity
  - Budget
- **ANN algorithm matters more than branding**

---

# When to Use What

## Practical decision framework for Vector DBs

This section connects:
- Scale
- Latency
- Cost
- Metadata
- Team maturity

**No theory. Only real-world choices.**

---

## 9️⃣4️⃣ First question: How big is your data?

### A) Small (≤ 100k vectors)

**Use cases:**
- Experiments
- Learning RAG
- Local tools

**✅ Best choices:**
- Chroma
- PostgreSQL + pgvector
- FAISS Flat

**❌ Avoid:**
- Milvus
- Pinecone

### B) Medium (100k – 10M vectors)

**Use cases:**
- Internal tools
- Knowledge bases
- AI assistants

**✅ Best choices:**
- FAISS (HNSW / IVF)
- Chroma (upper bound)
- pgvector (carefully)
- Weaviate

### C) Large (10M – 1B+ vectors)

**Use cases:**
- Search platforms
- Customer-facing RAG
- Enterprise AI

**✅ Best choices:**
- Pinecone
- Milvus
- FAISS + custom infra

---

## 9️⃣5️⃣ Second question: Do you need metadata filtering?

### ❌ Minimal metadata (Just similarity search)

**Use:**
- FAISS
- Pinecone (basic filters)

### ✅ Heavy metadata filters

**Examples:**
- Source type
- Time range
- Severity
- Component ID

**Use:**
- Chroma
- Weaviate
- Milvus
- pgvector

📌 **RAG without metadata degrades fast.**

---

## 9️⃣6️⃣ Third question: Production or experimentation?

### Experimentation / learning

**Goals:**
- Fast iteration
- Easy debugging

**Use:**
- Chroma
- pgvector
- FAISS Flat

### Production systems

**Goals:**
- SLA (service level agreement)
- Uptime
- Scalability

**Use:**
- Pinecone
- Milvus
- Weaviate Cloud

---

## 9️⃣7️⃣ Fourth question: Infra & Ops maturity

### Low infra maturity
- Small team
- No SRE (site reliability engineering)

**Use:**
- Pinecone
- Managed Weaviate

### High infra maturity
- Kubernetes
- Monitoring
- On-call

**Use:**
- Milvus
- FAISS-based services
- Self-hosted Weaviate

---

## 9️⃣8️⃣ Fifth question: Latency requirements

| Latency Target | Recommendation |
|---|---|
| <10 ms | FAISS HNSW |
| 10–50 ms | Pinecone / Weaviate |
| 50–200 ms | Chroma / pgvector |
| Batch | FAISS GPU |

---

## 9️⃣9️⃣ Golden rules (don't violate these)

### Rule 1️⃣
One embedding model per collection / index. Mixing models = meaningless similarity.

### Rule 2️⃣
Normalize if using cosine similarity. Always.

### Rule 3️⃣
Index choice matters more than DB brand. Bad index → bad RAG.

### Rule 4️⃣
Retrieval quality > model size.
A smaller LLM + good retrieval beats a bigger LLM + bad retrieval.

---

## 1️⃣0️⃣0️⃣ Typical real-world stacks

### Startup RAG
```
Embeddings → Pinecone → GPT
```

### Internal enterprise RAG
```
Embeddings → Weaviate → Re-ranker → LLM
```

### Research / offline
```
Embeddings → FAISS → Analysis
```

### SQL-heavy org
```
Embeddings → pgvector → LLM
```

---

**End of Complete LLM Fundamentals Guide**
