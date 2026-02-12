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
PHASE 4: Embeddings & RAG

Goal: Make LLMs “know things”

1️⃣1️⃣ Embeddings

What vectors are

Why cosine similarity

1️⃣2️⃣ Vector databases

FAISS

Chroma

When to use which

1️⃣3️⃣ RAG flow

Chunking

Retrieval

Augmented prompting
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
PHASE 5: Agents (RAIN / AIRA level)

Goal: Production AI systems

1️⃣4️⃣ What is an agent

Tools vs agents

Decision loops

1️⃣5️⃣ Tool calling

Function schemas

Controlled outputs

1️⃣6️⃣ Memory types

Short-term (context)

Long-term (vector DB)

PHASE 6: Production & System Design

Goal: Real-world readiness

1️⃣7️⃣ FastAPI + LLM

API wrappers

Streaming via SSE

1️⃣8️⃣ Security

API keys

Environment variables

1️⃣9️⃣ Cost & performance

Tokens = money

Latency tradeoffs

2️⃣0️⃣ Evaluation & logging

Prompt versioning

Observability