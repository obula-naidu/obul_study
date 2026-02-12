1️⃣ What is an LLM

A neural network trained to predict the next token based on previous tokens.
“An LLM is a neural network trained to predict the next token in a sequence; tokens are the atomic units of text the model processes.”

What an LLM is NOT

 It does not “understand” like humans
 It does not store memory
 It does not know facts inherently
 It does not reason like logic engines

Input tokens
   ↓
Neural network
   ↓
Probability distribution of next token
   ↓
Pick next token
   ↓
Repeat


Tokens (not words)

Tokens are NOT words
Tokens are Pieces of text the model actually processes

One word ≠ one token
One token ≠ one word

Why tokens exist

LLMs:
Cannot process raw characters efficiently
Cannot process entire words reliably

So text is converted into tokens using a tokenizer.

Tokens control:
🔹 Cost (for cloud LLMs)
More tokens = more money
🔹 Context window
Models have a maximum number of tokens

Example:
Context window = 8,000 tokens

That includes:
System messages
User messages
Assistant replies

🔹 Speed
More tokens = slower response

Input tokens vs Output tokens

When you send:
Explain FastAPI in detail

You pay/use tokens for:
Input tokens (your prompt)
Output tokens (model’s answer)

Example breakdown

Prompt:
Explain FastAPI
≈ 3 tokens
Response:
FastAPI is a modern, fast web framework...
≈ 50 tokens

Total = ~53 tokens

Tokenization demo (conceptual)

Text:
ChatGPT is helpful

Becomes:
[1345, 2987, 203, 9876]

The model never sees words, only numbers.

Why LLMs feel “smart”

Because:
They’ve seen trillions of token patterns
They predict the most likely continuation
They generate fluent language

But under the hood:
It’s next-token prediction, nothing more.

Stateless nature
LLMs do not store memory
Chat APIs do not keep history
Every request is independent

“LLMs are stateless; conversational memory is simulated by resending context.”

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
2️⃣ Chat vs Generate 

Why chat “feels” stateful

Who stores memory

3️⃣ Message roles

system / user / assistant

Why system messages matter
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Student analogy

Parameters → brain size (like llama3.2b here 3.2billion are parameters) 
        - “3B parameters means the model has about 3 billion learned numerical weights that operate on tokenized inputs; they are not the tokens themselves.”
Training tokens → books read
Context window → short-term memory
Tokens → words in a conversation

A student can:
Read millions of books
But still has one brain


Quantization is the process of reducing the precision of a model’s parameters to make LLMs smaller, faster, and cheaper to run.
Key points
LLMs have billions of parameters (numbers)
Normally stored as 32-bit floats (FP32) → very large memory
Quantization stores them using fewer bits (16, 8, or 4)

Why it’s done

✅ Reduces RAM / VRAM usage
✅ Enables local inference (laptops, CPUs)
✅ Faster loading and execution
⚠️ Slight quality loss (usually minor)

| Type | Bits per parameter | Approx memory |
| ---- | ------------------ | ------------- |
| FP32 | 32                 | ~12.8 GB      |
| FP16 | 16                 | ~6.4 GB       |
| INT8 | 8                  | ~3.2 GB       |
| Q4   | 4                  | ~1.6 GB       |
This is why Ollama models are often 2–4 GB, not 13 GB.

What quantization affects
✔ Memory
✔ Speed
✔ Deployment feasibility

What it does not affect
❌ Context window
❌ Training data
❌ Model architecture

One-line takeaway (memorize this)
Quantization compresses model weights by reducing numerical precision, allowing large LLMs to run efficiently with minimal quality loss.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
4️⃣ Context window

Context window is the maximum number of tokens an LLM can see at one time in a single request.

Key points
*Measured in tokens, not words
*Includes:
    - System messages
    - User messages
    - Assistant replies
*Prompt + response together must fit inside it

What happens if it’s exceeded
*Oldest tokens are dropped (truncation), or
*Conversation is summarized, or
*Request fails (depends on system)

Why it exists
*LLMs use attention, which is computationally expensiv
*More tokens = more memory + slower inference

Important clarifications
❌ Tokens ≠ context window
✔ Context window = capacity limit
✔ Tokens = content filling that limit

Why models “forget”
*LLMs are stateless
*When tokens exceed the context window, earlier messages fall out
*System instructions can also be lost if too long

One-line takeaway (memorize this)
The context window is the maximum number of tokens an LLM can attend to in a single request; exceeding it causes loss of earlier context.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
5️⃣ Decoding parameters(temperature,top_p,max_tokens,stop)

Prompt
  ↓
Tokenization
  ↓
MODEL (3.2B learned weights)
  ↓
Token probabilities
  ↓
Decoding parameters (temperature, top_p, etc.)
  ↓
Chosen token
  ↓
Repeat


Decoding parameters are runtime controls that decide how an LLM selects the next token, without changing the model’s learned knowledge.

Model parameters = what the model knows
Decoding parameters = how it chooses words

What they are
*Applied after the model computes token probabilities
*Do not modify model weights
*Affect style, randomness, and length, not intelligence

| Parameter             | What it controls                       |
| --------------------- | -------------------------------------- |
| **temperature**       | Randomness of output                   |
| **top_p**             | Limits choices to most probable tokens |
| **max_tokens**        | Maximum tokens in response             |
| **stop**              | Tokens that end generation             |
| **presence_penalty**  | Encourages new topics                  |
| **frequency_penalty** | Reduces repetition                     |
| **seed**              | Makes output reproducible              |

1.Temperature — Randomness of output

What it actually does
*Controls how sharp or flat the probability distribution is
*Applied before sampling

Intuition
*Low temperature → model picks the most likely token
*High temperature → model explores less likely tokens

| Temperature | Behavior               |
| ----------- | ---------------------- |
| 0.0–0.2     | Deterministic, factual |
| 0.3–0.7     | Balanced               |
| 0.8–1.2     | Creative               |
| >1.5        | Chaotic / nonsense     |

temperature = 0 ≈ greedy decoding (no randomness)
Does not add new knowledge

2.top_p (nucleus sampling) — Limits token choices

The nucleus is the smallest set of next-token candidates whose combined probability mass reaches top_p
At a given step, the model predicts probabilities for the next token only (not future tokens)
From that:
Tokens are sorted by probability
Added one by one
Stop when cumulative probability ≥ top_p
That resulting set = nucleus

What it actually does
*Chooses the smallest set of tokens whose cumulative probability ≥ top_p
*Sampling happens only inside this set

Intuition
*Prevents low-probability garbage tokens
*Keeps responses coherent

If probabilities are:
A: 0.50
B: 0.25
C: 0.15
D: 0.05
E: 0.05
-top_p = 0.9 → {A, B, C}
-top_p = 0.6 → {A, B}

| top_p | Behavior     |
| ----- | ------------ |
| 0.9   | Safe default |
| 0.7   | Conservative |
| 0.5   | Very strict  |

**top_k limits the model to choosing only the K most probable next tokens, regardless of their probabilities.(rarely used)
**Temperature reshapes probabilities; top_p selects from them — so temperature must come first.

3.max_tokens — Response length limit

What it actually does

*Hard upper bound on generated tokens
*Includes only the output, not prompt

Why it exists
*Prevents infinite generatio
*Controls cost & latency

Important clarifications
*If model finishes early → stops naturally
*If limit is reached → output is cut off

4.stop — When generation must end

What it actually does
*Defines exact token sequences that force termination
*Checked after each token

Common use cases
*End at "User:"
*End at "###"

Stop before leaking system prompts
Example
"stop": ["\n\n", "User:"]

When model outputs any of these → generation halts

Important
Stop tokens are not included in output
Multiple stop sequences allowed



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


Using high temperature + high top_p → rambling
Very low max_tokens → incomplete answers
Forgetting stop → unwanted continuation


| Use case   | temperature | top_p | max_tokens |
| ---------- | ----------- | ----- | ---------- |
| Factual QA | 0.2         | 0.9   | 200        |
| Chatbot    | 0.6         | 0.9   | 300        |
| Creative   | 0.9         | 0.95  | 500        |
| Code       | 0.2         | 0.8   | 400        |

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
👉 After Phase 1, nothing about LLM APIs will feel magical.

PHASE 2: Ollama (Local LLMs)

Goal: Comfort with local models

6️⃣ Ollama architecture

Ollama is a local HTTP server that loads LLM model files and exposes them via simple REST APIs.

Ollama = Local OpenAI server on your machine

*Instead of api.openai.com - You have localhost:11434
*Instead of cloud GPUs     - You use your CPU / GPU

What happens when you run this
ollama run llama3.2

Internally:
Ollama checks if model exists
If not:
    Downloads compressed model (zstd)
    Decompresses it
Loads model into memory
Starts inference loop
Accepts prompts

*Ollama ≠ model
*Ollama = model runner + API server
*Model = .gguf file loaded by Ollama

Model files

Where Ollama stores models - Ollama stores models locally.
Typical locations: Linux: ~/.ollama/models
Inside, you’ll see files like: llama3.2-3b.Q4_K_M.gguf

What a .gguf file contains

A .gguf file is not just weights.
It includes:
-Model architecture (layers, heads)
-Learned weights (quantized)
-Tokenizer & vocab
-Context window size
-Metadata

📌 Once loaded, Ollama does not need the internet.

zstd IS:

A compression format like zip
Used when downloading models - compress large model and decompress and store as gguf

Ollama downloads compressed models (zstd), stores them as quantized .gguf files, and loads them into memory for inference.

CPU vs GPU inference
Inference = Using already-trained weights to predict the next token

CPU inference (default)
How it works:

Uses highly optimized C++ (llama.cpp)
Uses:
    SIMD
    AVX / AVX2 / AVX512
Runs on normal CPU cores

GPU Inference
How it works:

Moves heavy matrix multiplications to GPU
Uses:
    CUDA (NVIDIA)
    Metal (Mac)
    ROCm (limited)

Ollama:
    Auto-detects GPU
    Automatically offloads layers
    Falls back to CPU if needed
You don’t manually choose CPU/GPU in most cases.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

7️⃣ Ollama APIs - refer to llm-basics.md

/api/chat - Multi-turn conversation,Role-based messages

/api/generate - Single prompt → single response

/api/embeddings - text → vectors

/api/tags - List installed models along with its metadata
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

8️⃣ Streaming responses - check in llm_basics.py 19-40

Token-by-token streaming
The model sends the response token by token (or chunk by chunk) instead of waiting for the full answer.
Without streaming:
    Tokens are generated
    Buffered
    Sent only after completion

With streaming:
    Tokens are sent as soon as they are generated

Why it matters for UX:
    Instant feedback
    Feels fast & alive
    Essential for chat appss
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

PHASE 3: OpenAI-style APIs (Cloud LLMs)

Goal: Switch providers without confusion
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
9️⃣ OpenAI ChatCompletion-style APIs

Same concepts, different URLs
only:
    URLs change
    Auth is added
    Limits exist
Ollama APIs and OpenAI APIs are conceptually the same.


Rate limits
Rate limits (new constraint)
Cloud models are shared.

Limits exist on:
    Requests per minute
    Tokens per minute
If exceeded → errors (next task).

| Limit | Meaning             |
| ----- | ------------------- |
| RPM   | Requests per minute |
| TPM   | Tokens per minute   |


API keys
Why API keys exist
    Identify user
    Enforce billing
    Apply rate limits
How they are used
Authorization: Bearer sk-xxxx

Cloud LLM APIs do not change how LLMs work — they only add authentication, billing, and limits.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔟 Error handling & retries

Timeouts
A timeout means:
“The server didn’t respond within the time I’m willing to wait.”
| Type               | Where          |
| ------------------ | -------------- |
| Connection timeout | Network / DNS  |
| Read timeout       | Model is slow  |
| Client timeout     | Your SDK limit |


429, errors

Let’s say your limits are:
60 RPM
90,000 TPM

You send:
10 requests
Each = 10,000 tokens (input + output)
❌ 100,000 tokens → 429 error
Even though RPM is OK.

What triggers 429 (Rate Limit)?
Common causes
    Too many parallel requests
    Large prompts
    Streaming many tokens
    Agent loops

{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Too many requests"
  }
}

Bad code (naive)

for query in queries:
    call_llm(query)

Exponential backoff
Good code (with retry + backoff)
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

Why exponential backoff works

If everyone retries immediately → thundering herd
1s → 2s → 4s → 8s → success
Allows:

Token bucket to refill
Queue to clear


500-level errors = server-side failures
| Code | Meaning             |
| ---- | ------------------- |
| 500  | Internal error      |
| 502  | Bad gateway         |
| 503  | Service unavailable |
| 504  | Gateway timeout     |

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

| Error   | Retry? | Delay              |
| ------- | ------ | ------------------ |
| Timeout | ✅      | Exponential        |
| 500     | ✅      | Exponential        |
| 502/503 | ✅      | Longer             |
| 429     | ✅      | Provider-specified |
| 400     | ❌      | Fix request        |
| 401     | ❌      | Fix auth           |



--------------------------------------------------------------------------------------------------------------------------------------------------------------------
PART A — Embeddings (Foundation)

11.1 What embeddings are and token vs embeddings
An embedding is a numerical vector that represents meaning.

Not text.
Not tokens.
Meaning.

Why embeddings exist

LLMs:

Generate text
Are bad at searching large knowledge bases

Embeddings:
    Convert text → numbers
    Enable similarity search

Example (conceptual)
"FastAPI is a Python framework"
→ [0.13, -0.44, 0.82, ..., 0.09]  (1536 dimensions)
These vectors are close in space.

What vectors actually encode

They encode:
    Topic
    Intent
    Semantics
    Relationships

They do ❌ NOT encode:
    Grammar
    Exact wording
    Order (mostly)

| Concept    | Purpose            |
| ---------- | ------------------ |
| Tokens     | Generation         |
| Embeddings | Search & retrieval |

Embeddings turn meaning into geometry.

❌ Embeddings are not model parameters
❌ Embeddings are not learned per query
❌ Embeddings are not context windows
They are outputs of a trained embedding model.


Where embeddings live in RAG
Docs → embeddings → vector DB
Query → embedding → similarity search
Top chunks → prompt → LLM

Where embeddings live in RAG
Docs → embeddings → vector DB
Query → embedding → similarity search
Top chunks → prompt → LLM

Why embeddings make LLMs “know things”
LLMs:
    Don’t store your documents
RAG:
    Retrieves relevant docs at runtime
    Injects them into prompt

Knowledge is externalized

11.2 Embedding models and Dimensionality

What is an embedding model?
An embedding model is a neural network trained to map text → vectors such that semantic similarity = geometric closeness.
📌 Different from a chat/generation model.

How embedding models are trained (conceptual)

They are trained on:
    Sentence pairs
    Question–answer pairs
    Paraphrases
    Contrastive learning

Training goal:
    similar meaning → vectors close
    different meaning → vectors far

Types of embedding models
🔹 Proprietary (Cloud)
    OpenAI text-embedding-3-large
    Cohere
    Google

🔹 Open-source
    SentenceTransformers
    BGE (BAAI)
    E5
    GTE
    Instructor models

🔹 Local (Ollama)
    nomic-embed-text
    mxbai-embed-large
    bge-base

Dimensionality
| Model        | Dimensions |
| ------------ | ---------- |
| OpenAI small | 768        |
| OpenAI large | 3072       |
| BGE-base     | 768        |
| BGE-large    | 1024       |
| nomic        | 768        |

| Lower dim   | Higher dim    |
| ----------- | ------------- |
| Faster      | More accurate |
| Less memory | Better nuance |
| Cheaper     | Slower        |
768–1024 is the industry sweet spot.

One crucial rule (people mess this up)
Query and documents MUST use the same embedding model.
Mixing models = broken similarity search.

Embedding normalization

Most models output vectors that are:
    Already normalized OR
    Should be normalized
Why?
    Makes cosine similarity stable
    Improves indexing
Many vector DBs auto-normalize.

When embeddings FAIL
    Very short queries (“yes”, “ok”)
    Exact keyword search
    Numbers / IDs
    Highly structured data

Use hybrid search (later topic).

Mental model
    Embedding model = semantic encoder
    Vector DB = memory
    LLM = reasoning engine

11.4 Similarity metrics (cosine, dot, L2)
Why similarity metrics exist
Once you have embeddings (vectors), you need to answer:
    “How close are these two meanings?”
    That’s what similarity metrics do.

1️⃣ The three main similarity metrics
| Metric                  | Used for       |
| ----------------------- | -------------- |
| Cosine similarity       | Most common    |
| Dot product             | Fast, ranking  |
| L2 (Euclidean) distance | Geometry-based |

Cosine similarity (most important)
Measures the angle between two vectors, not their length.

Why cosine is king 👑
Ignores magnitude
Focuses on direction (meaning)
Stable across embedding models
Works well with normalized vectors

📌 Most embedding models are trained expecting cosine similarity.

| Cosine value | Meaning           |
| ------------ | ----------------- |
| 1.0          | Identical meaning |
| 0.8          | Very similar      |
| 0.5          | Somewhat related  |
| 0.0          | Unrelated         |
| -1.0         | Opposite meaning  |

Dot product
Measures both direction and magnitude

When dot product is used
    Vectors are normalized
    Speed is critical
    Ranking is more important than exact similarity

📌 If vectors are normalized:
dot product ≈ cosine similarity
That’s why some DBs use dot product internally.

2 (Euclidean) distance
Straight-line distance between vectors

Why L2 is less popular
    Sensitive to magnitude
    Worse semantic behavior
    Less aligned with training objectives

Used mostly in:
    Vision models
    Older embedding systems

Why cosine works best for text

Text meaning is:
    Directional
    Relative
    Scale-independent

Cosine captures exactly that.

Vector DB perspective

Most vector DBs support all metrics, but:
| DB       | Default       |
| -------- | ------------- |
| FAISS    | Inner product |
| Chroma   | Cosine        |
| Pinecone | Cosine        |
| Weaviate | Cosine        |

Common mistake (critical)

❌ Using cosine on non-normalized vectors
❌ Mixing similarity metrics between indexing & querying

📌 Index metric == query metric (must match).

Meaning = direction
Similarity = angle
Cosine = angle comparison

What “dimension” really means
    One dimension = one learned semantic feature
An embedding of size 768 means:
    768 independent semantic signals
Not human-interpretable, but statistically meaningful.

Why embeddings have FIXED size
Neural networks require:
    Fixed input size
    Fixed output size
So:
    Any text length → same-size vector

That’s why:
1 sentence
1 paragraph
1 page
All become 768 numbers (for that model).

Why not 10 dimensions? Why not 1 million?
Too few dimensions
    Can’t represent nuance
    Many meanings collapse together
    Poor retrieval quality

Too many dimensions
    Slow search
    High memory
    Harder indexing
    Diminishing returns

Why 768 / 1024 became standard
These numbers come from:
    Transformer hidden sizes
    Powers of 2
    Hardware efficiency

Example:

| Model family | Hidden size |
| ------------ | ----------- |
| BERT-base    | 768         |
| RoBERTa      | 768         |
| BGE-base     | 768         |
| BGE-large    | 1024        |


📌 Embedding head often mirrors hidden size.

Curse of dimensionality (important)

As dimensions increase:
    Distance between points becomes less meaningful
    Everything starts to look “far”
    Indexing gets harder

Vector DBs combat this with:
    Approximate nearest neighbor (ANN)
    Quantization
    Clustering

Practical tradeoffs
| Use case              | Recommended dims |
| --------------------- | ---------------- |
| Small app             | 384–768          |
| RAG systems           | 768–1024         |
| High-precision search | 1024–1536        |
| Edge / mobile         | ≤384             |

Can you reduce dimensions?
Yes:
    PCA
    Autoencoders
    Quantization

But:
❌ usually hurts retrieval
✔️ useful for memory-constrained systems

Mental model

Dimensions = semantic resolution
More dims = sharper meaning
Fewer dims = blurrier meaning

When Embeddings FAIL
This task explains why RAG systems sometimes give bad answers even with embeddings.

Embeddings are semantic, not factual.
They capture meaning similarity, not truth or exactness.

Major failure cases
1. Very short queries
Examples:
“yes”
“ok”
“why?”
➡️ Too little semantic signal
➡️ Vectors are noisy
Fix: Expand query or use conversation context.

2. Keyword-heavy queries
Examples:
    Error codes (ERR_CONN_RESET)
    IDs (order_839201)
    File names
Embeddings blur exact tokens.
Fix: Keyword search or hybrid search.

3. Numerical & tabular data
Examples:
    Prices
    Dates
    Metrics
Embeddings don’t preserve numeric precision.
Fix: Structured DB + RAG.

4. Domain mismatch
Embedding model not trained on:
    Telecom logs
    Kernel traces
    Medical codes
➡️ Similarity becomes meaningless.
Fix: Domain-specific embeddings.

5. Long documents, bad chunking
    Important info split across chunks
    Context lost
Fix: Smarter chunking (next section).

6. False positives (semantic drift)
Query:
“How to reset router?”
Retrieved:
“How to restart application”

Semantically similar but wrong.

Why embeddings don’t “understand”
They optimize:
similar meaning → close vectors

They do NOT optimize:
    Logical correctness
    Temporal truth
    Causality
That’s LLM’s job.

4️⃣ Warning sign in RAG

If your RAG answers:
    Confidently wrong
    With irrelevant citations
➡️ Retrieval is broken, not generation.

Embeddings vs search engines
| Feature     | Embeddings | Keyword search |
| ----------- | ---------- | -------------- |
| Semantic    | ✅          | ❌              |
| Exact match | ❌          | ✅              |
| Numbers     | ❌          | ✅              |
| Speed       | Medium     | Fast           |

➡️ That’s why hybrid search exists.

 Mental model
Embeddings = fuzzy semantic lens
Keyword search = sharp literal lens

Best systems use both.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔜 NEXT PHASE — PART B
🧩 TASK 12.1 — Vector Databases (What & Why)

Why we need vector DBs

Why SQL isn’t enough

ANN indexing basics

Say “next” to continue.



PART B — Vector Databases

12️⃣ Vector stores
12.1 Indexing methods
12.2 FAISS (in-memory, performance)
12.3 Chroma (metadata + dev)
12.4 Alternatives (Pinecone, Weaviate, Milvus, pgvector)
12.5 When to use what

PART C — RAG (Core System)

13️⃣ RAG architecture
13.1 Chunking strategies
13.2 Retrieval strategies
13.3 Augmented prompting
13.4 Context window budgeting
13.5 Failure modes

PART D — Advanced RAG (Production)

14️⃣ Improvements & alternatives
14.1 Hybrid search
14.2 Re-ranking
14.3 Multi-query RAG
14.4 Agentic RAG
14.5 RAG vs Fine-tuning
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
PHASE 5: Agents (RAIN / AIRA level)

Goal: Production AI systems

1️⃣5️⃣ What is an agent

Tools vs agents

Decision loops

1️⃣6️⃣ Tool calling

Function schemas

Controlled outputs

1️⃣7️⃣ Memory types

Short-term (context)

Long-term (vector DB)

PHASE 6: Production & System Design

Goal: Real-world readiness

1️⃣8️⃣ FastAPI + LLM

API wrappers

Streaming via SSE

1️⃣9️⃣ Security

API keys

Environment variables

2️⃣0️⃣ Cost & performance

Tokens = money

Latency tradeoffs

2️⃣1️⃣ Evaluation & logging

Prompt versioning

Observability