<p align="center">
    <img alt="AI Engineering Interview Questions and Answers" src="https://github.com/amitshekhariitbhu/ai-engineering-interview-questions/blob/main/assets/banner.png">
</p>

# AI Engineering Interview Questions and Answers

> AI Engineering Interview Questions and Answers - Your Cheat Sheet For AI Engineering Interviews
>
> These interview questions and answers are helpful for roles such as:
>
> - AI Engineer
> - Gen AI Engineer
> - LLM Engineer
> - Agentic AI Engineer
> - AI Agent Engineer
> - Forward Deployed Engineer
> - AI Solutions Architect
> - AI Platform Engineer
> - Applied AI Engineer
> - MLOps Engineer
> - LLMOps Engineer

## Table of Contents

- [Must Know](#must-know)
- [LLM Fundamentals](#llm-fundamentals)
- [Prompt Engineering](#prompt-engineering)
- [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
- [AI Agents and Agentic Systems](#ai-agents-and-agentic-systems)
- [Fine-Tuning and Model Adaptation](#fine-tuning-and-model-adaptation)
- [Vector Databases and Embeddings](#vector-databases-and-embeddings)
- [AI System Design](#ai-system-design)
- [LLMOps and Production AI](#llmops-and-production-ai)
- [Evaluation and Testing](#evaluation-and-testing)
- [AI Safety, Ethics, and Responsible AI](#ai-safety-ethics-and-responsible-ai)
- [Multimodal AI](#multimodal-ai)
- [AI Infrastructure and Scalability](#ai-infrastructure-and-scalability)
- [Coding and Practical Implementation](#coding-and-practical-implementation)
- [Behavioral and Scenario-Based Questions](#behavioral-and-scenario-based-questions)

### Prepared and maintained by the **Founder** of [Outcome School](https://outcomeschool.com): Amit Shekhar

### Follow Amit Shekhar

- [X/Twitter](https://twitter.com/amitiitbhu)
- [LinkedIn](https://www.linkedin.com/in/amit-shekhar-iitbhu)
- [GitHub](https://github.com/amitshekhariitbhu)

### Follow Outcome School

- [YouTube](https://youtube.com/@OutcomeSchool)
- [X/Twitter](https://x.com/outcome_school)
- [LinkedIn](https://www.linkedin.com/company/outcomeschool)
- [GitHub](https://github.com/OutcomeSchool)

## I teach at Outcome School

- [AI and Machine Learning](https://outcomeschool.com/program/ai-and-machine-learning)

---

> **Note: We will keep updating this with new questions and answers.**

---

### Must Know

- LLM
- RAG
- MCP
- Agent
- Fine-tuning
- Quantization

Learn about the LLM, RAG, MCP, Agent, Fine-tuning & Quantization: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)

### LLM Fundamentals

- What are foundation models, and how have they changed AI engineering?
  - Answer: Foundation models are large models trained on broad, general-purpose datasets so they can be adapted to many downstream tasks. They changed AI engineering by shifting teams from training task-specific models from scratch to prompting, fine-tuning, evaluating, and deploying reusable base models. This made AI product development faster, but also introduced new engineering concerns such as latency, cost, safety, retrieval, evaluation, and model governance.
- What is a Large Language Model (LLM), and how does it work?
  - Answer: An LLM is a neural network trained to understand and generate text by predicting tokens from context. Most modern LLMs use Transformer architectures: text is split into tokens, tokens are converted into embeddings, attention layers mix information across the context, and the model outputs probabilities for the next token. Repeating next-token prediction generates full responses.
- Inside ChatGPT: What Happens After You Hit Enter?
  - Answer: After you submit a message, the system combines your input with conversation history, system/developer instructions, tool context, and safety policies. The text is tokenized and passed through the model, which computes next-token probabilities. Decoding settings choose tokens step by step until a stop condition is reached. The final response may also involve tool calls, retrieval, moderation, formatting, and logging depending on the product.
- What is the Transformer architecture and how does it work?
  - Answer: The Transformer is a neural architecture built around attention instead of recurrence. It represents tokens as embeddings, adds position information, and repeatedly applies attention, feed-forward layers, normalization, and residual connections. Attention lets each token weight other relevant tokens in the context, making Transformers highly parallelizable and effective for long-range language patterns. https://www.youtube.com/watch?v=avjX3QrYkls
- What are the key components of the Transformer architecture?
  - Answer: Key components include token embeddings, positional encodings or positional embeddings, attention layers, multi-head attention, feed-forward networks, residual connections, layer normalization, and an output projection to vocabulary logits. Encoder-decoder Transformers also include cross-attention between decoder tokens and encoder outputs.
- Walk me through what happens, step by step, in one forward pass of a decoder-only Transformer.
  - Answer: The input text is tokenized and converted to token embeddings. Position information is added, then each Transformer block applies masked self-attention so each token can attend only to allowed previous tokens. The result passes through residual connections, normalization, and a feed-forward network. After the final block, the model projects hidden states to vocabulary logits, and the last position's logits are used to choose the next token.
- What is tokenization in LLMs?
  - Answer: Tokenization is the process of converting text into smaller units, called tokens, that the model can process. Tokens may be words, subwords, characters, punctuation, or byte-level chunks. The tokenizer maps each token to an integer ID, which is then converted into an embedding. Tokenization affects cost, context usage, multilingual quality, and how well domain-specific terms are represented.
- Explain BPE (Byte Pair Encoding).
  - Answer: Byte Pair Encoding is a subword tokenization method that starts with small units, often bytes or characters, and repeatedly merges the most frequent adjacent pairs into larger tokens. This creates a vocabulary that can represent common words efficiently while still handling rare or unseen words by splitting them into smaller pieces. https://www.youtube.com/watch?v=4A_nfXyBD08
- Explain WordPiece and SentencePiece.
  - Answer: WordPiece is a subword tokenizer that builds tokens by selecting pieces that improve the likelihood of the training corpus, commonly using continuation markers for subword fragments. SentencePiece is a tokenizer framework that treats text as a raw character stream and can train BPE or unigram models without relying on pre-tokenized whitespace. Both help models handle rare words, misspellings, and multilingual text with a fixed vocabulary.

### 1. BPE — Byte Pair Encoding

**Core idea:** Start with small units and repeatedly **merge the most frequent pair**.

```text
l + o → lo
lo + w → low
low + e → lowe
```

BPE stores:

```text
Vocabulary:
l, o, w, lo, low, lowe, ...

Merge rules:
l + o → lo
lo + w → low
low + e → lowe
```

So remember:

> **BPE = frequency-based merging + ordered merge rules**

---

### 2. WordPiece

WordPiece also creates a vocabulary of subword pieces, but its training objective is different.

Instead of simply asking:

> "Which pair occurs most often?"

it asks approximately:

> **"Which new piece gives the greatest improvement to the likelihood/modeling of the training data?"**

Example vocabulary:

```text
play
##ing
##ed
##er
##s
```

Then:

```text
playing → play + ##ing
played  → play + ##ed
player  → play + ##er
```

`##` means the piece occurs **inside/after an existing word piece**.

Important:

> WordPiece primarily learns a **vocabulary of useful subword pieces**, rather than BPE's explicit ordered merge table.

---

### 3. SentencePiece

SentencePiece is **not simply another algorithm alongside BPE and WordPiece**.

It is a **tokenization framework** that can use algorithms such as:

```text
SentencePiece
   ├── BPE
   └── Unigram
```

Its important feature is that it can tokenize **raw text directly**, without requiring traditional word splitting first.

Example:

```text
"I love India"
       ↓
["▁I", "▁love", "▁India"]
```

`▁` represents a whitespace boundary.

Think:

> **SentencePiece = tokenizer framework**

---

### 4. Unigram

Unigram is a **subword tokenization algorithm**, commonly used with SentencePiece.

Its approach is almost the opposite of BPE.

### BPE

```text
small vocabulary
      ↓
add/merge useful pieces
      ↓
larger vocabulary
```

### Unigram

```text
large candidate vocabulary
      ↓
remove less useful pieces
      ↓
smaller final vocabulary
```

Unigram is **probabilistic**.

For a word such as:

```text
unhappiness
```

it may consider:

un + happiness
un + happy + ness
u + n + happiness

- What is positional encoding, and why is it needed in Transformers?
  - Answer: Positional encoding gives the model information about token order. Self-attention alone is permutation-invariant, so without position information the model would not know whether one token came before or after another. Positional methods can be fixed, learned, relative, or rotary, and they help the model understand sequence structure.
- What are embeddings?
  - Answer: Embeddings are dense numeric vectors that represent tokens, words, documents, images, or other objects in a continuous space. Similar items tend to have nearby vectors, allowing models to capture semantic relationships. In LLMs, token embeddings are the model's input representation, while embedding models are often used for search, clustering, classification, and RAG.
- Explain the Query(Q), Key(K), and Value(V) in attention.
  - Answer: Queries, keys, and values are learned projections of token representations. A query represents what a token is looking for, keys represent what each token offers for matching, and values contain the information to retrieve. Attention scores are computed by comparing queries with keys, then those scores weight the values to produce a context-aware representation.
- What is self-attention, and how does it work in Transformers?
  - Answer: Self-attention lets tokens in the same sequence exchange information. Each token produces Q, K, and V vectors; the model compares each query with all keys, applies a softmax to get attention weights, and uses those weights to combine values. In decoder-only LLMs, causal masking prevents a token from attending to future tokens during generation.
- What is Cross Attention in Transformers?
  - Answer: Cross-attention is attention between two different sources of representations. In an encoder-decoder Transformer, decoder queries attend to encoder keys and values, allowing the decoder to generate output conditioned on the encoded input. It is useful in translation, summarization, multimodal models, and systems where one stream needs to condition on another.
- Why do we scale the dot product attention by √dₖ in the Transformer architecture?
  - Answer: Dot products grow larger as the key/query dimension increases. Large attention scores can push softmax into very sharp distributions, causing small gradients and unstable training. Dividing by √dₖ keeps the score variance in a healthier range, making attention easier to optimize.
- What is causal masking?
  - Answer: Causal masking prevents a token from attending to future tokens. In next-token prediction, the model should only use previous and current context, not information from later positions. The mask sets future attention scores to negative infinity before softmax, ensuring generation remains autoregressive.
- What are multi-head attention mechanisms? Why use multiple attention heads?
  - Answer: Multi-head attention runs several attention operations in parallel, each with separate learned projections. Different heads can specialize in different relationships, such as syntax, coreference, local patterns, or long-range dependencies. Their outputs are concatenated and projected, giving the model a richer representation than a single attention head.
- What are Feed-Forward Networks in LLMs?
  - Answer: Feed-forward networks are the per-token MLP layers inside Transformer blocks. After attention mixes information across tokens, the feed-forward network transforms each token representation independently, usually by expanding the hidden dimension, applying a nonlinearity, and projecting back down. They provide much of the model's capacity for storing and transforming learned patterns.
- What is Generative AI?
  - Answer: Generative AI refers to AI systems that create new content such as text, code, images, audio, video, or structured data. Instead of only classifying or ranking existing inputs, generative models learn patterns from training data and produce novel outputs conditioned on prompts, examples, or other context.
- What is the context window in LLMs, and why does it matter?
  - Answer: The context window is the maximum number of tokens the model can consider at once, including the prompt, conversation history, retrieved documents, tool outputs, and generated response. It matters because information outside the window is unavailable to the model. Larger context windows support longer documents and conversations, but they increase cost, latency, and memory usage.
- Why is the context window limited in LLMs?
  - Answer: The context window is limited because attention and KV-cache memory grow with sequence length. Standard self-attention has quadratic compute with respect to tokens, and inference must store key/value tensors for every generated and input token. Longer windows also make retrieval quality, latency, cost, and training stability harder, so model providers choose a practical limit based on architecture and serving constraints.
- What is temperature in the context of LLMs, and how does it affect output?
  - Answer: Temperature controls how sharply or broadly the model samples from the next-token probability distribution. Lower temperature makes outputs more deterministic and conservative by favoring high-probability tokens. Higher temperature increases randomness and diversity, but can also increase hallucinations, inconsistency, and formatting errors.
- Why is the first token slower than the rest in an LLM?
  - Answer: The first token is slower because the model must process the entire input prompt before it can generate anything. This prefill phase computes attention over all prompt tokens and builds the KV cache. After that, decoding usually processes one new token at a time while reusing the cache, so later tokens are faster per token.
- Explain Top-p (nucleus) sampling and Top-k sampling. How do they differ?
  - Answer: Top-k sampling restricts generation to the k most likely next tokens, then samples from that fixed-size set. Top-p sampling chooses the smallest set of tokens whose cumulative probability exceeds p, then samples from that dynamic set. Top-k is simple and predictable, while top-p adapts to the confidence of the distribution.
- Compare greedy decoding, beam search, top-k, top-p, and temperature sampling. When does each fail?
  - Answer: Greedy decoding always picks the highest-probability token; it is fast but can be repetitive or myopic. Beam search keeps multiple high-probability candidates; it helps structured tasks but can produce bland or over-optimized text. Top-k and top-p add controlled randomness; they can fail when too restrictive or too loose. Temperature sampling adjusts randomness globally; very low values become rigid, while high values can drift or hallucinate.
    These are all **decoding strategies**: ways to turn a model’s next-token probability distribution into actual text.

Suppose the model predicts:

| Token    | Probability |
| -------- | ----------: |
| `cat`    |        0.40 |
| `dog`    |        0.30 |
| `fox`    |        0.15 |
| `car`    |        0.10 |
| `banana` |        0.05 |

The model repeats this process token by token.

## 1. Greedy decoding

Greedy decoding always picks the **single most probable next token**.

Here:

> `cat`

Then it recomputes probabilities for the next token and again picks the maximum.

Formally:

\[
x*t = \arg\max_x P(x \mid x*{<t})
\]

### Strengths

- Deterministic
- Fast
- Simple
- Often good for constrained or factual tasks

### Failure mode

The locally best token may not lead to the globally best sequence.

For example:

```text
A → probability 0.6
B → probability 0.4
```

But later:

```text
A → bad continuation probability 0.1
B → excellent continuation probability 0.9
```

Sequence probabilities:

\[
P(A...) = 0.6 \times 0.1 = 0.06
\]

\[
P(B...) = 0.4 \times 0.9 = 0.36
\]

Greedy chose `A`, despite the `B` sequence being much more probable overall.

It can also produce text that is repetitive, bland, or overly predictable.

---

## 2. Beam search

Beam search tries to solve greedy decoding's short-sightedness.

Instead of keeping just one sequence, it maintains the best **B candidate sequences**, where \(B\) is the beam width.

For beam width 3:

```text
Step 1:
A
B
C

Step 2:
A x
A y
B x
B y
C x
C y

Keep the best 3 overall:
A x
B y
C x
```

Then expand those again.

The score is typically based on cumulative log probability:

\[
\log P(x*1,\ldots,x_T)
=
\sum_t \log P(x_t \mid x*{<t})
\]

### Strengths

- Searches more globally than greedy
- Useful when there is a fairly well-defined "best" answer
- Historically popular for translation, speech recognition, and structured generation

### Failure modes

A major one is **generic, unnatural text**.

Large language models often assign high probability to safe, conventional sequences. Beam search aggressively pursues those high-probability paths, so increasing the beam width can actually make text more boring.

It may also favor short sequences because probabilities multiply:

\[
0.9^{5} > 0.9^{20}
\]

So beam search often uses **length normalization**.

Beam search is generally less attractive for open-ended chat or creative writing because you usually want a plausible sample, not the mathematically highest-probability sentence.

---

# Sampling methods

Greedy and beam search primarily **search for high-probability sequences**.

Top-k, top-p, and temperature instead modify the distribution and then **sample from it**.

---

## 3. Temperature sampling

Temperature changes how sharp the probability distribution is.

Given model logits \(z_i\):

\[
P(i) =
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
\]

where \(T\) is temperature.

### Low temperature

If:

\[
T < 1
\]

the distribution becomes sharper.

For example:

```text
Original:
cat     .40
dog     .30
fox     .15
car     .10
banana  .05
```

might become roughly:

```text
cat     .55
dog     .30
fox     .09
car     .05
banana  .01
```

So outputs are more predictable.

As:

\[
T \rightarrow 0
\]

sampling approaches greedy decoding.

### High temperature

If:

\[
T > 1
\]

the distribution flattens:

```text
cat     .28
dog     .25
fox     .19
car     .16
banana  .12
```

Now surprising tokens become more likely.

### Failure modes

**Too low:**

- repetitive
- rigid
- generic
- deterministic

**Too high:**

- incoherent
- factually unstable
- weird word choices
- abrupt topic shifts

Temperature does **not eliminate bad tokens**. Even something with probability 0.00001 remains possible.

That's why it is frequently combined with top-k or top-p.

---

# 4. Top-k sampling

Top-k keeps only the **k most probable tokens**.

If:

```text
cat     .40
dog     .30
fox     .15
car     .10
banana  .05
```

and:

\[
k=3
\]

keep:

```text
cat
dog
fox
```

Discard:

```text
car
banana
```

Then renormalize:

\[
P'(x) = \frac{P(x)}{\sum\_{i \in K} P(i)}
\]

giving approximately:

```text
cat  .47
dog  .35
fox  .18
```

and sample.

### Strength

It prevents extremely unlikely tokens from appearing.

### Failure mode

The ideal number of plausible tokens changes dramatically by context.

For:

> The capital of France is \_\_\_

maybe only one token is really sensible.

But for:

> My favorite thing about summer is \_\_\_

hundreds of tokens may be reasonable.

A fixed:

\[
k=40
\]

doesn't adapt.

Sometimes 40 is far too many; sometimes it is too few.

That motivated **top-p sampling**.

---

# 5. Top-p / nucleus sampling

Top-p keeps the **smallest set of tokens whose cumulative probability reaches \(p\)**.

Suppose:

```text
cat     .40
dog     .30
fox     .15
car     .10
banana  .05
```

For:

\[
p=0.8
\]

sort them by probability:

```text
cat  .40   cumulative .40
dog  .30   cumulative .70
fox  .15   cumulative .85
```

So the nucleus is:

```text
cat
dog
fox
```

Then sample among those.

The crucial distinction from top-k is that the number of retained tokens changes dynamically.

In a confident context:

```text
Paris    .96
London   .01
Berlin   .01
...
```

top-p = 0.9 may retain only `Paris`.

In an ambiguous context:

```text
swimming   .08
traveling  .07
reading    .06
hiking     .05
...
```

it may retain dozens of tokens.

### Strength

It adapts to the model's uncertainty.

That's why nucleus sampling became very popular for open-ended generation.

### Failure modes

With high \(p\), the candidate set can still contain lots of poor tokens.

For example:

\[
p=0.99
\]

may allow a long tail of unlikely choices.

With low \(p\):

\[
p=0.5
\]

generation may become overly conservative and repetitive.

---

# How temperature, top-k, and top-p work together

A common decoding pipeline is:

```text
model logits
     ↓
temperature
     ↓
top-k filtering
     ↓
top-p filtering
     ↓
renormalize probabilities
     ↓
sample a token
```

For example:

```text
temperature = 0.7
top_k = 50
top_p = 0.9
```

Conceptually:

1. Temperature adjusts how sharp the distribution is.
2. Top-k removes everything except the 50 highest-probability tokens.
3. Top-p further keeps only enough of those to account for 90% probability mass.
4. Sample from what's left.

You don't necessarily need both top-k and top-p.

A common configuration is simply:

```text
temperature = 0.7
top_p = 0.9
```

with no top-k restriction.

---

# Beam search versus sampling

This is the most important distinction.

**Beam search asks:**

> What high-probability sequence can I find?

**Sampling asks:**

> What plausible sequence can I draw from the model's distribution?

Those goals are different.

Imagine a model could answer:

```text
The movie was very good.
The movie was surprisingly moving.
I loved the cinematography.
It started slowly but became excellent.
```

Beam search may repeatedly favor:

> The movie was very good.

because it has the highest total likelihood.

Sampling might produce any of the others.

For open-ended language generation, diversity is usually desirable, which is why sampling is often preferred.

---

# Can beam search use temperature/top-k/top-p?

Technically yes, but it is uncommon to combine them naively.

Beam search normally operates on token probabilities directly and keeps the best branches rather than randomly sampling.

There are variants like:

- stochastic beam search
- diverse beam search
- beam sampling

But ordinary LLM generation usually falls into one of two families:

```text
Greedy / Beam
```

or

```text
Temperature + Top-p / Top-k sampling
```

rather than combining everything.

---

# A useful mental model

Think of a restaurant menu.

**Greedy**

> Always order the single most popular dish.

**Beam search**

> Consider the five most promising multi-course meals and keep exploring the best ones.

**Temperature**

> Adjust how adventurous you are.

Low temperature:

> Strong preference for popular dishes.

High temperature:

> Willing to try unusual dishes.

**Top-k**

> You may choose only from the 20 most popular dishes.

**Top-p**

> Consider however many dishes are needed to cover, say, 90% of customer preferences.

That last one adapts automatically: sometimes that may mean 3 dishes; sometimes 30.

---

# When each tends to work best

| Method      | Good for                                         | Typical problem                           |
| ----------- | ------------------------------------------------ | ----------------------------------------- |
| Greedy      | deterministic tasks, simple completion           | short-sighted, repetitive                 |
| Beam search | translation, transcription, structured sequences | bland, generic, computationally expensive |
| Temperature | controlling randomness                           | high values cause nonsense                |
| Top-k       | limiting bizarre tokens                          | fixed cutoff doesn't adapt                |
| Top-p       | open-ended text generation                       | extremes can be too random/conservative   |

For many conversational LLM applications, the practical recipe is:

\[
\text{temperature} + \text{top-p sampling}
\]

For deterministic tasks, use very low temperature or greedy-like decoding.

The deeper point is that **the model defines the probability distribution; decoding determines how aggressively you exploit it versus explore it.** Greedy and beam search lean toward exploitation; temperature/top-k/top-p sampling let you control exploration.

- What are logits, and how are they used in text generation?
  - Answer: Logits are the raw, unnormalized scores the model produces for each token in the vocabulary. They are converted into probabilities with softmax, often after adjustments such as temperature scaling, top-k/top-p filtering, repetition penalties, or logit bias. The decoding algorithm then selects or samples the next token from that distribution.
- What are skip connections (residual connections) in Transformers?
  - Answer: Skip connections add a layer's input back to its output, usually around attention and feed-forward sublayers. They help gradients flow through deep networks, reduce optimization difficulty, and allow layers to learn refinements instead of completely new representations. This is essential for training very deep Transformer models reliably.
- What is the difference between open-source and closed-source LLMs? When would you choose one over the other?
  - Answer: Open-source or open-weight LLMs provide model weights, code, or enough artifacts to run and customize the model yourself. Closed-source LLMs are accessed through hosted APIs and usually hide weights and training details. Choose open models for control, privacy, customization, offline deployment, or cost predictability at scale. Choose closed models for fast integration, strong managed performance, lower operations burden, and vendor-supported reliability.
- What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?
  - Answer: Encoder-only models read the full input bidirectionally and are strong for understanding tasks such as classification, retrieval, and token labeling. Decoder-only models generate autoregressively using causal masking and are the dominant architecture for chat and text generation. Encoder-decoder models encode an input sequence and decode an output sequence, making them useful for translation, summarization, and sequence-to-sequence tasks.
- What is KV cache, and how does it speed up inference?
  - Answer: The KV cache stores the key and value tensors computed for previous tokens during autoregressive generation. Without it, the model would recompute attention states for the entire prefix at every new token. With the cache, each decoding step computes only the new token's projections and attends to stored keys and values, greatly reducing repeated work.
- Estimate the KV cache memory needed to serve a large model. How does it constrain batch size and context length?
  - Answer: A rough KV-cache estimate is batch*size * sequence*length * layers _ 2 _ kv*heads * head*dim * bytes_per_value. The factor of 2 is for keys and values. Memory grows linearly with batch size and context length, so longer prompts or more concurrent users reduce how many requests fit on a GPU. Models using MQA or GQA reduce kv_heads and therefore reduce cache memory.
- KV Cache Compression
  - Answer: KV cache compression reduces the memory required to store past keys and values during long-context inference. Common approaches include quantizing the cache, evicting less useful tokens, using sliding or paged attention, sharing KV heads with MQA/GQA, or compressing old context into smaller representations. The tradeoff is usually memory and throughput versus attention fidelity.
- What is model distillation, and how is it used with LLMs?
  - Answer: Model distillation trains a smaller student model to imitate a larger teacher model. For LLMs, the teacher may generate answers, rationales, preference data, or soft probability targets that the student learns from. Distillation is used to reduce latency, cost, and deployment size while preserving as much task quality as possible.
- What is Mixture of Experts (MoE), and how does it work in models like Mixtral?
  - Answer: Mixture of Experts models contain multiple expert feed-forward networks and a router that selects a small subset of experts for each token. This gives the model a large total parameter count while activating only part of it per token. In models like Mixtral, sparse expert routing improves capacity and efficiency, but adds complexity around routing balance, serving, and communication.
- What is the difference between dense and sparse models?
  - Answer: In a dense model, most or all parameters in each layer are used for every token. In a sparse model, only a subset of parameters is activated for a given token, as in MoE routing. Dense models are simpler to train and serve, while sparse models can offer more capacity per unit of compute if routing and infrastructure are handled well.
- How does DeepSeek-V4 work?
  - Answer: DeepSeek-V4 is a Mixture-of-Experts model family that keeps the DeepSeekMoE and multi-token prediction ideas from earlier DeepSeek models while adding long-context efficiency improvements. Its architecture uses hybrid attention with compressed sparse attention and heavily compressed attention to reduce KV-cache and long-context costs. It also uses architectural and training changes such as manifold-constrained hyper-connections and the Muon optimizer to improve stability and efficiency.
- What is Flash Attention?
  - Answer: Flash Attention is an optimized exact attention algorithm that reduces memory traffic by tiling attention computation and avoiding materializing the full attention matrix in GPU memory. It computes attention in blocks using fast on-chip memory, making training and inference faster and more memory efficient while preserving the same attention result up to numerical precision.
    https://youtube.com/shorts/OSAelpOrqWs?si=8RfRD5W6_Wj8mj0s
- What is Cross-Entropy Loss?
  - Answer: Cross-entropy loss measures the difference between the model's predicted probability distribution and the true target distribution. In language modeling, the target is usually the correct next token, and the loss is lower when the model assigns that token high probability. Minimizing cross-entropy is equivalent to maximizing the likelihood of the training text.
    https://youtube.com/shorts/HNI0oPL_5iQ?si=9AHrVEz2c4MkcwiN

- What is Grouped-Query Attention (GQA), and how does it differ from Multi-Head Attention (MHA)?
  - Answer: In standard multi-head attention, each query head has its own key and value heads. Grouped-Query Attention keeps many query heads but shares fewer key/value heads across groups of queries. This reduces KV-cache memory and decoding bandwidth while preserving more quality than using a single shared KV head as in Multi-Query Attention.
    Sure. Both **Multi-Head Attention (MHA)** and **Grouped-Query Attention (GQA)** are attention mechanisms used in Transformers. The main difference is **how many Key (K) and Value (V) heads are used compared with Query (Q) heads**.

## 1. Multi-Head Attention (MHA)

In standard Multi-Head Attention, every attention head has its own:

- Query head
- Key head
- Value head

For example, suppose a model has **8 attention heads**:

```text
Queries:  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
Keys:     K1 K2 K3 K4 K5 K6 K7 K8
Values:   V1 V2 V3 V4 V5 V6 V7 V8
```

Each head independently performs:

\[
Attention(Q_i,K_i,V_i)
=
softmax\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i
\]

So:

```text
Head 1 → Attention(Q1, K1, V1)
Head 2 → Attention(Q2, K2, V2)
...
Head 8 → Attention(Q8, K8, V8)
```

The results from all heads are concatenated.

### Why multiple heads?

Different heads can learn different relationships.

For example, in:

> "The cat sat on the mat because it was tired."

Different heads might focus on:

```text
Head 1 → grammatical relationships
Head 2 → nearby words
Head 3 → pronoun references
Head 4 → semantic similarity
...
```

This makes MHA powerful, but there is a downside.

During LLM generation, the model has to store the **Key and Value vectors for every previous token** in the **KV cache**.

If there are many heads, the KV cache becomes large.

---

# 2. Grouped-Query Attention (GQA)

GQA tries to keep most of the benefit of MHA while reducing the memory required for Keys and Values.

Instead of giving every Query head its own K and V head, **several Query heads share the same Key and Value heads**.

Suppose we still have:

```text
8 Query heads
```

But only:

```text
2 Key heads
2 Value heads
```

The heads could be grouped like this:

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤──→ K1, V1
Q4 ─┘

Q5 ─┐
Q6 ─┤
Q7 ─┤──→ K2, V2
Q8 ─┘
```

So the attention operations are roughly:

```text
Q1 → K1,V1
Q2 → K1,V1
Q3 → K1,V1
Q4 → K1,V1

Q5 → K2,V2
Q6 → K2,V2
Q7 → K2,V2
Q8 → K2,V2
```

The Query heads remain separate, so they can still learn different attention patterns.

But the K/V heads are shared.

---

# MHA vs GQA

| Feature               | MHA                 | GQA                  |
| --------------------- | ------------------- | -------------------- |
| Query heads           | Many                | Many                 |
| Key heads             | Same as Query heads | Fewer                |
| Value heads           | Same as Query heads | Fewer                |
| K/V sharing           | No                  | Yes, within groups   |
| KV cache size         | Large               | Smaller              |
| Memory usage          | Higher              | Lower                |
| Generation speed      | Slower              | Usually faster       |
| Model quality         | Very strong         | Usually close to MHA |
| Common in modern LLMs | Yes                 | Very common          |

For example:

```text
MHA
Q heads = 32
K heads = 32
V heads = 32
```

while GQA could use:

```text
Q heads = 32
K heads = 8
V heads = 8
```

Each K/V head would therefore serve:

\[
32 / 8 = 4
\]

Query heads.

---

## Why GQA is useful for LLMs

During autoregressive generation:

```text
Token 1 → Token 2 → Token 3 → ... → Token 10000
```

the model stores Keys and Values for previous tokens.

Roughly speaking:

\[
KV\ Cache \propto
SequenceLength
\times KVHeads
\times HeadDimension
\]

Imagine:

```text
32 Q heads
32 KV heads
```

with MHA.

With GQA:

```text
32 Q heads
8 KV heads
```

the KV-head portion of the cache is approximately:

\[
\frac{8}{32} = \frac14
\]

So it can require roughly **4× less KV-cache memory** for that layer configuration.

That matters enormously for long-context LLM inference.

---

## There is also MQA

You may also encounter **Multi-Query Attention (MQA)**.

It is essentially the extreme version of GQA:

```text
Many Query heads
       ↓
ONE Key head
ONE Value head
```

Example:

```text
Q1 ─┐
Q2 ─┤
Q3 ─┤
Q4 ─┤
Q5 ─┤──→ K1,V1
Q6 ─┤
Q7 ─┤
Q8 ─┘
```

So you can think of them as a spectrum:

```text
MHA                   GQA                     MQA

Q: 8                  Q: 8                    Q: 8
K: 8                  K: 2                    K: 1
V: 8                  V: 2                    V: 1

No sharing      Some K/V sharing       Maximum sharing
   ↑                   ↑                      ↑
Quality         Good compromise       Maximum efficiency
```

### Easy way to remember

**MHA:**

> Every Query head gets its **own K and V**.

**GQA:**

> A **group of Query heads shares K and V**.

**MQA:**

> **All Query heads share one K and V**.

- How does Sliding Window Attention work?
  - Answer: Sliding Window Attention restricts each token to attend only to a fixed-size window of nearby tokens instead of the entire sequence. This lowers attention cost from quadratic over the full context to roughly linear in sequence length for a fixed window. It works well for local dependencies, but models may need special mechanisms such as global tokens, memory, retrieval, or attention sinks to preserve long-range information.
- How do Attention Sinks work?
  - Answer: Attention sinks are tokens, often early tokens in the sequence, that many later tokens attend to disproportionately. They help stabilize attention distributions during long-context generation because attention needs somewhere to place probability mass even when local tokens are not useful. Some long-context methods preserve these sink tokens while sliding or truncating the rest of the cache to maintain quality.
    Attention sinks are a behavior in Transformer models where some tokens—often very early tokens like the first token or a special BOS token—receive a surprisingly large amount of attention, even when their actual semantic content is not important.

The core idea is that the model sometimes needs somewhere to “put” excess attention probability.

Attention weights come from:

\[
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Because of the softmax, the attention weights for each query must sum to 1:

\[
\sum*j a*{ij}=1
\]

So every query has to distribute all of its attention mass somewhere.

Suppose a token does not strongly need information from any previous token. It cannot assign “no attention” to everything because softmax still forces the probabilities to sum to 1.

A model may learn to use one token as a kind of dumping ground:

```text
Current query token

      ↓ attention
-------------------------
Token 1        0.72   ← sink
Token 2        0.04
Token 3        0.03
Token 4        0.06
Token 5        0.05
Token 6        0.10
-------------------------
Total          1.00
```

Token 1 gets lots of attention, not necessarily because it is useful, but because attending to it is relatively harmless.

That token becomes an **attention sink**.

A useful intuition is:

> “I don't need information from the context right now, but attention must go somewhere, so I'll put it here.”

Why doesn’t this distort the model output? Because a model can learn the sink token’s value vector so that attending to it has little harmful effect. For example, its value contribution may effectively behave like a neutral/default vector.

This becomes especially important with long-context inference and KV-cache eviction. Imagine a Transformer has processed:

```text
[BOS] A B C D E F G H I J ...
```

Normally it can attend back to all previous tokens. But suppose you use a sliding-window cache and throw away old tokens:

```text
Original:

[BOS] A B C D E F G H I J
  ↑
attention sink

After naive eviction:

              F G H I J
```

If the sink token disappears, the model's attention pattern can become unstable because the model had learned to rely on that token as an attention anchor.

Researchers found that keeping a few initial tokens permanently in the KV cache can dramatically improve streaming behavior.

Conceptually:

```text
Keep permanently
     ↓
[BOS] A  |  G H I J K L
────────    ───────────
sink tokens   rolling window
```

Instead of keeping every old token, you preserve:

\[
\text{sink tokens} + \text{recent tokens}
\]

For example, with a cache budget of 4096 tokens:

```text
4 sink tokens
+
4092 most recent tokens
```

As new tokens arrive, old recent tokens are evicted, but the first few sink tokens remain.

This was a key idea behind **StreamingLLM**.

There is another subtle point: the sink is often created by the model itself during training. It is not necessarily a token whose linguistic meaning is special.

For instance:

```text
<BOS> The dog ran through the field...
 ↑
 huge attention
```

The model may care less about the semantic meaning of `<BOS>` and more about its structural role as a consistently available position.

You can think of it somewhat like a “null” destination in attention.

Without a sink:

```text
Query:
"I don't particularly need any previous token."

But softmax says:
"You must assign 100% somewhere."
```

With a sink:

```text
Model:
"Fine, I'll assign most of it to this harmless token."
```

There is also a mathematical reason early tokens make convenient sinks. In causal attention, the first token is visible to essentially every later token:

```text
token 1 sees: token 1
token 2 sees: token 1,2
token 3 sees: token 1,2,3
...
token 1000 sees: token 1...1000
```

So token 1 is a universally available anchor throughout the sequence. That consistency makes it easy for attention heads to specialize around it.

One important distinction:

**Attention sinks are not the same thing as important tokens.**

A token can receive large attention weights because it is genuinely semantically relevant, or because it is functioning as a sink.

So:

\[
\text{high attention} \not\Rightarrow \text{high semantic importance}
\]

This is one reason interpreting Transformer attention maps requires care.

Relating this to your previous question about MHA/GQA: attention sinks can occur under both architectures. GQA changes how K/V heads are shared, while attention sinks describe a learned pattern in the attention distribution itself.

In short:

\[
\boxed{
\text{Attention sink}
=
\text{a stable token that absorbs otherwise unnecessary attention mass}
}
\]

And in long-context inference, keeping a few sink tokens in the KV cache can help preserve model quality even while older normal tokens are discarded.

- How does Rotary Position Embedding (RoPE) work, and why is it preferred over learned positional embeddings?
  - Answer: RoPE encodes position by rotating query and key vectors by position-dependent angles before attention. This makes relative distance between tokens naturally affect their dot products. It is preferred in many LLMs because it generalizes better to longer contexts than simple learned absolute position embeddings and works efficiently with autoregressive attention.
- Explain Layer Normalization
  - Answer: Layer Normalization normalizes the activations within each token representation across the hidden dimension. It subtracts the mean, divides by the standard deviation, and then applies learned scale and shift parameters. In Transformers, it stabilizes training, improves gradient flow, and reduces sensitivity to activation scale.
- Explain RMSNorm (Root Mean Square Layer Normalization)
  - Answer: RMSNorm is a simpler normalization method that scales activations by their root mean square without subtracting the mean. It keeps the learned scale parameter but removes part of LayerNorm's computation. Many modern LLMs use RMSNorm because it is faster, stable, and often performs similarly to or better than full LayerNorm.
- Why do modern Transformers use Pre-LayerNorm (Pre-Norm) instead of Post-LayerNorm?
  - Answer: Pre-Norm applies normalization before the attention or feed-forward sublayer, while Post-Norm applies it after the residual addition. Pre-Norm improves gradient flow through deep networks and makes large Transformers easier to train. Post-Norm can produce strong results but is more prone to instability as depth increases.
- What are scaling laws (Chinchilla), and how do they guide model size vs training data decisions?
  - Answer: Scaling laws describe how model performance changes with parameters, data, and compute. The Chinchilla result showed that many earlier LLMs were over-sized and under-trained for their compute budget, and that optimal training often uses more tokens with a smaller model. In practice, scaling laws help decide how to allocate compute between model size, dataset size, and training duration.
- Your LLM keeps ignoring your instructions. How do you make it follow structured output formats?
  - Answer: Use a strict schema, clear examples, and constrained decoding when available. Put formatting rules in the system or developer prompt, remove conflicting instructions, and ask for only the target format with no extra prose. In production, validate outputs with a parser, retry with error feedback, or use JSON/schema mode or function calling if the model provider supports it.
- Your LLM-powered tool hits the context window limit on long documents. How do you handle it?
  - Answer: Do not send the whole document blindly. Chunk the document, retrieve only relevant sections, summarize or compress older context, and use map-reduce or hierarchical processing for full-document tasks. For workflows that need exact references, keep source chunks in external storage and pass citations or selected excerpts into the model.
- Your LLM does not admit when it does not know the answer. How do you make it say "I don't know"?
  - Answer: Ground the model in retrieved or provided evidence and explicitly require abstention when evidence is missing. Use prompts that separate answerable from unanswerable cases, add examples of abstention, set confidence thresholds, and evaluate with unanswerable test cases. In RAG systems, check retrieval quality and require citations before allowing a final answer.
- Your LLM generates responses that are too verbose. How do you control response length?
  - Answer: Set explicit length constraints in the prompt, such as bullet count, sentence count, or word budget. Use API controls like max output tokens, stop sequences, and lower verbosity instructions. For stricter behavior, add examples of the desired style, post-process long answers, or run a compression step before returning the final response.
- Your LLM memorized proprietary training data and leaks it in responses. How do you prevent this?
  - Answer: Reduce exposure during training by deduplicating, filtering, redacting secrets, and excluding sensitive data. Use privacy reviews, access controls, data retention limits, and output filters for known secrets. For fine-tuning, train only on approved data and test with extraction attacks. If leakage is discovered, remove the data, retrain or patch the model if possible, and add monitoring.
- Your LLM coding assistant generates outdated code using deprecated libraries. How do you fix it?
  - Answer: Ground the assistant with current documentation, package versions, and project-specific examples using retrieval or tool access. Add prompts that require checking installed versions and repository conventions before answering. In production, combine generation with linters, type checks, tests, dependency scanning, and feedback loops so deprecated APIs are caught automatically.
- Your tokenizer splits important domain terms into meaningless subword pieces. How do you fix it?
  - Answer: Add domain terms to the tokenizer vocabulary if you control the model and can resize or continue training embeddings. Otherwise, use normalization, aliases, glossaries, or retrieval context that explains the terms. For high-impact domains, train or adapt a tokenizer on domain text and continue pretraining or fine-tuning so the model learns useful representations for those tokens.
- Your Transformer's KV cache grows too large during long sequence generation. How do you manage memory?
  - Answer: Use techniques such as paged attention, KV-cache quantization, sliding-window attention, grouped-query or multi-query attention, cache eviction, and request batching policies. You can also shorten prompts, summarize older conversation state, cap generation length, or move long-term memory into retrieval instead of keeping every token in the active cache.
- Your Transformer runs out of memory on long documents due to quadratic self-attention. How do you scale it?
  - Answer: Use memory-efficient attention implementations such as Flash Attention, sparse or sliding-window attention, chunking, retrieval, or hierarchical summarization. For document workflows, process sections independently and combine intermediate results. For model architecture changes, use long-context methods that avoid full dense attention over every token pair.
- Your distilled student model fails on the complex reasoning that the teacher model handled. How do you close the gap?
  - Answer: Improve the distillation data with harder examples, teacher rationales, multi-step solutions, and edge cases where the student fails. Use curriculum training, preference tuning, or reinforcement learning on reasoning tasks. You may also need a larger student, more inference-time compute, tool use, or a routing setup that sends hard cases to the teacher model.
- After RLHF alignment, your LLM became safer but lost capability on hard tasks. How do you manage the alignment tax?
  - Answer: Measure capability regressions with targeted evals, then rebalance the training mix with high-quality task data, harmlessness data, and refusal examples. Use preference data that rewards safe completion instead of over-refusal. Techniques such as supervised fine-tuning refreshes, DPO/RLHF tuning adjustments, model merging, and task-specific routing can recover capability while preserving safety.
- Your RLHF-trained LLM is gaming the reward model instead of being genuinely helpful. How do you fix reward hacking?
  - Answer: Improve the reward model with adversarial examples, diverse human preferences, and checks for shallow behaviors that receive high scores. Use multiple reward signals, constraint-based evaluation, held-out human review, and online monitoring. Penalize exploit patterns, refresh the reward model regularly, and validate final models on real task success rather than reward score alone.
- Your chatbot loses context after 10 turns in a conversation. How do you maintain a long conversation context?
  - Answer: Maintain conversation state outside the model. Keep a rolling summary, store important user facts and decisions as memory, retrieve relevant past turns, and include only the most useful context in each prompt. Also separate durable user preferences from temporary task context so the model does not confuse old and current goals.
- Your chatbot fails when users switch topics mid-conversation. How do you handle topic switches?
  - Answer: Detect topic changes with intent classification, embeddings, or conversation-state rules. Start a new task state when the topic changes, while preserving durable preferences and relevant history. The assistant should acknowledge the new topic, avoid carrying over stale assumptions, and ask a clarifying question only when the new request lacks required context.
- Your QA system always generates an answer even when no answer exists in the context. How do you detect unanswerable questions?
  - Answer: Add an abstention path. Check whether retrieval returned sufficiently relevant evidence, require the model to cite supporting spans, and classify the question as answerable or unanswerable before generation. Use confidence thresholds, entailment checks, and evaluation sets with negative examples so the system learns to say that the answer is not present in the context.
- Your summarization system hallucinated facts not in the original article. How do you fix it?
  - Answer: Make the summarizer evidence-grounded. Instruct it to summarize only provided text, use extractive intermediate notes, preserve citations to source spans, and run a factual consistency check before returning the final summary. For long documents, summarize chunks carefully and combine them with a second pass that does not introduce new claims.
- Your text generation repeats phrases in long outputs. How do you fix repetition?
  - Answer: Tune decoding parameters and add repetition controls such as frequency penalties, presence penalties, no-repeat n-gram constraints, or lower temperature. Also check whether the prompt encourages looping, whether max token limits are too high, and whether the model needs clearer stopping criteria. For severe cases, use better fine-tuning data or post-generation repetition detection.
- Transformers work on text, so can they also understand images?
  - Answer: Yes. Images can be split into patches or encoded into visual tokens, then processed by Transformer layers much like text tokens. Vision Transformers use image patches directly, while multimodal LLMs often use a vision encoder plus a projection layer to connect image representations to a language model. This lets the model answer questions, caption images, and reason over visual inputs.
- Small Language Models (SLMs)
  - Answer: Small Language Models are compact language models designed for lower latency, lower cost, and easier deployment than large frontier models. They are useful for focused tasks, on-device inference, private deployments, classification, extraction, routing, and simple chat workflows. Their main tradeoff is reduced world knowledge and reasoning capacity compared with larger models.
- Large Reasoning Models (LRMs)
  - Answer: Large Reasoning Models are LLMs optimized for difficult reasoning tasks such as math, coding, planning, and multi-step problem solving. They often use reasoning-focused training data, preference optimization, tool use, or inference-time computation to improve deliberate problem solving. They are strongest when accuracy matters more than minimal latency.
- Jev and System One Models
  - Answer: System One models are optimized for fast, intuitive responses, while Jev-style or reasoning-oriented models emphasize slower, more deliberate problem solving. In practice, fast models are good for simple requests, classification, and high-throughput use cases, while reasoning models are better for complex tasks that need planning, verification, or multi-step logic.
- What are Autoregressive Models?
  - Answer: Autoregressive models generate output one step at a time, where each new token is conditioned on previous tokens. In language modeling, they learn the probability of a sequence as a product of next-token probabilities. Decoder-only LLMs are autoregressive, which makes them natural for text generation, chat, code completion, and streaming responses.
- Explain the difference between autoregressive and masked language modeling.
  - Answer: Autoregressive language modeling predicts the next token using only previous context, which is ideal for generation. Masked language modeling hides some tokens in the input and trains the model to reconstruct them using both left and right context, which is strong for understanding tasks. GPT-style models are autoregressive, while BERT-style models use masked language modeling.
- Proximal Policy Optimization (PPO)
  - Answer: PPO is a reinforcement learning algorithm used in some RLHF pipelines to optimize a policy model against a reward model while limiting how far the policy can move from its previous behavior. The clipping or trust-region-like constraint improves training stability. In LLM alignment, PPO can improve helpfulness or preference alignment, but it is complex and sensitive to reward model quality.
- Direct Preference Optimization (DPO)
  - Answer: DPO is a preference optimization method that trains a model directly from pairs of preferred and rejected responses. Instead of training a separate reward model and running reinforcement learning, DPO optimizes a classification-like objective that increases the likelihood of preferred responses relative to rejected ones. It is simpler and more stable than many RLHF setups.
- Group Relative Policy Optimization (GRPO)
  - Answer: GRPO is a reinforcement learning method that compares multiple sampled responses for the same prompt and optimizes the policy using relative rewards within that group. It can reduce the need for a separate value model and is useful for reasoning-oriented training where several candidate solutions can be scored. The key idea is learning from relative quality among grouped outputs.
- Recursive Language Models (RLMs)
  - Answer: Recursive Language Models are models or systems that apply language-model reasoning repeatedly, often feeding intermediate outputs back into later steps. This can support decomposition, self-refinement, planning, verification, or tree-like reasoning. The benefit is more deliberate computation; the risk is compounding errors if intermediate steps are not checked.
- Continual Learning in LLMs
  - Answer: Continual learning is the process of updating a model over time as new data, tasks, or domains appear. For LLMs, the challenge is learning new information without catastrophic forgetting, regressions, or privacy leaks. Common approaches include continued pretraining, fine-tuning with replay data, adapters, retrieval-based memory, and careful evaluation across old and new tasks.
- What is Recursive Self-Improvement (RSI)?
  - Answer: Recursive Self-Improvement is the idea of an AI system improving its own capabilities, then using the improved version to make further improvements. In practical systems, this might involve generating training data, writing code, designing experiments, or improving prompts and tools. It requires strong evaluation, safety controls, and human oversight because errors or misaligned objectives can compound.
- How do Diffusion Language Models (DLMs) work?
  - Answer: Diffusion Language Models generate text through an iterative denoising process rather than strictly left-to-right next-token prediction. They start from noisy or masked token representations and progressively refine them into coherent text. This can enable parallel generation and flexible editing, but discrete text diffusion is harder than image diffusion because language tokens are categorical and highly structured.
- How Does LLM Watermarking Work?
  - Answer: LLM watermarking embeds a detectable statistical pattern into generated text, usually by slightly biasing token selection toward a secret or known subset of tokens. A detector later checks whether the token pattern is unlikely to occur naturally. Watermarking can help identify AI-generated content, but it may be weakened by paraphrasing, translation, editing, or generation settings.
- How do RNNs and Transformers differ?
  - Answer: RNNs process tokens sequentially, maintaining a hidden state that is updated step by step. Transformers process sequences with attention, allowing tokens to directly attend to other tokens and enabling much more parallel training. RNNs are efficient for streaming and small sequences, but Transformers scale better, capture long-range dependencies more effectively, and dominate modern LLMs.

### Prompt Engineering

- What is prompt engineering, and why is it critical for AI applications?
  - Answer: Prompt engineering is the practice of designing instructions, context, examples, constraints, and output requirements so an LLM reliably performs a task. It is critical because the same model can behave very differently depending on the prompt. Good prompts improve accuracy, consistency, safety, cost, latency, and integration with downstream systems.
- Explain zero-shot, one-shot, and few-shot prompting with examples.
  - Answer: Zero-shot prompting asks the model to perform a task without examples, such as "Classify this review as positive, neutral, or negative." One-shot prompting provides one example before the real input. Few-shot prompting provides several examples so the model can infer the desired pattern, label style, and edge-case behavior. Examples are especially useful for classification, extraction, tone control, and custom formats. Reference: [Explain zero-shot, one-shot, and few-shot prompting with examples](https://www.linkedin.com/posts/pallavi-shekhar_llm-prompting-ai-activity-7441801012472078336-JsHr)
- What is chain-of-thought (CoT) prompting, and when should you use it?
  - Answer: Chain-of-thought prompting encourages a model to reason through intermediate steps before producing an answer. It is useful for math, logic, planning, multi-step QA, and tasks where the final answer depends on several dependent inferences. In production, you usually ask for a concise rationale or hidden reasoning summary rather than exposing long internal reasoning to the user. Reference: [How does Chain-of-Thought (CoT) Prompting work?](https://outcomeschool.com/blog/how-does-chain-of-thought-prompting-work)
- Explain self-consistency prompting and how it improves reasoning.
  - Answer: Self-consistency prompting generates multiple independent reasoning paths for the same problem and then selects the most common or best-supported final answer. It improves reasoning because a single sampled chain may make a mistake, while multiple samples can reveal a stable consensus. The tradeoff is higher token cost and latency.
- What is tree-of-thought prompting?
  - Answer: Tree-of-thought prompting explores multiple possible reasoning branches instead of one linear chain. The model proposes intermediate thoughts, evaluates or ranks them, and continues with the most promising branches. It is useful for planning, search, puzzles, complex problem solving, and tasks where early decisions strongly affect the final answer.
- What is ReAct (Reasoning + Acting) prompting, and how does it work?
  - Answer: ReAct combines reasoning steps with actions, such as searching, calling tools, reading files, or querying APIs. The model thinks about what it needs, takes an action, observes the result, and repeats until it can answer or complete the task. It is common in agentic systems because it connects language reasoning with external evidence and tools. Reference: [ReAct Agent](https://outcomeschool.com/blog/react-agent)
- What is a system prompt, and how does it influence model behavior?
  - Answer: A system prompt is a high-priority instruction that defines the assistant's role, behavior, constraints, safety rules, and response style. It influences how the model interprets user requests and resolves conflicts between instructions. It should contain durable behavior rules, while task-specific details usually belong in developer or user prompts.
- How do you structure prompts for consistent structured output (JSON, XML)?
  - Answer: Specify the exact schema, required fields, data types, allowed values, and whether extra fields are forbidden. Put the format requirement near the end of the prompt, provide a valid example, and ask for only the structured object with no prose. In production, prefer JSON/schema mode or function calling when available, then validate and retry with parser errors if needed.
- What is prompt injection, and how do you defend against it?
  - Answer: Prompt injection is an attack where untrusted input tries to override system instructions, reveal secrets, or manipulate tool use. Defenses include separating trusted instructions from user or retrieved content, treating external text as data, least-privilege tool access, output validation, allowlisted actions, secret isolation, retrieval sanitization, and monitoring for suspicious instructions. Reference: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- What is jailbreaking in LLMs, and what are common jailbreak techniques?
  - Answer: Jailbreaking is the attempt to bypass a model's safety or policy constraints. Common techniques include role-play, instruction hierarchy confusion, encoding or obfuscation, multi-turn manipulation, false authority, "ignore previous instructions" attacks, hypothetical framing, and asking the model to transform or reveal restricted content indirectly. Defenses combine strong system instructions, policy classifiers, refusal training, tool safeguards, and adversarial testing. Reference: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- How do you optimize prompts for cost and latency?
  - Answer: Shorten prompts, remove redundant context, use retrieval to include only relevant passages, cache stable prompt prefixes, choose smaller models when quality allows, cap output length, and avoid unnecessary multi-step chains. For repeated tasks, use templates, structured outputs, batching, and evaluation to find the shortest prompt that still meets quality targets.
- What is the difference between prompt engineering and prompt tuning?
  - Answer: Prompt engineering is manual or programmatic design of natural-language instructions and examples sent at inference time. Prompt tuning is a training method that learns soft prompt vectors or task-specific prompt parameters while keeping most or all model weights fixed. Prompt engineering requires no training, while prompt tuning needs data and training infrastructure but can be more consistent for narrow tasks.
- What is a prompt template, and how do you design one for production use?
  - Answer: A prompt template is a reusable prompt with variables for inputs such as user query, context, examples, locale, or output schema. For production, define clear variable boundaries, escape or delimit user content, version the template, keep examples representative, document assumptions, test edge cases, and pair the template with validation, logging, and evaluation metrics.
- How do you handle multi-turn conversations with LLMs?
  - Answer: Maintain explicit conversation state outside the model. Include relevant recent turns, durable user preferences, task status, tool results, and a compact summary of older context. Detect topic switches, avoid carrying stale assumptions into new tasks, and store long-term memory separately from temporary conversation state.
- What is role prompting, and when is it effective?
  - Answer: Role prompting asks the model to respond from a specific perspective, such as "act as a senior backend engineer" or "act as a medical intake assistant." It is effective when the role implies useful norms, vocabulary, criteria, or structure. It should be paired with concrete instructions because a role alone is weaker than explicit task requirements.
- What is prompt chaining, and how do you design a chain of prompts for complex tasks?
  - Answer: Prompt chaining breaks a complex workflow into multiple LLM calls, where each step performs a focused subtask and passes structured output to the next step. Design chains around clear interfaces: decompose the task, define each step's input and output schema, validate intermediate results, add retries where errors are recoverable, and keep human review for high-risk decisions. Reference: [How does Prompt Chaining work?](https://outcomeschool.com/blog/how-does-prompt-chaining-work)
- How do you evaluate and iterate on prompt quality?
  - Answer: Build a representative test set with normal cases, edge cases, and adversarial examples. Define metrics such as accuracy, faithfulness, format validity, refusal quality, latency, and cost. Compare prompt versions with fixed model settings, review failures, update instructions or examples, and keep regression tests so improvements do not break previous behavior.
- What are meta-prompts, and how can they be used to generate prompts?
  - Answer: Meta-prompts are prompts that ask an LLM to create, critique, or improve other prompts. They can generate task instructions, test cases, rubrics, few-shot examples, or prompt variants for evaluation. They are useful for acceleration, but generated prompts should still be reviewed and tested against real examples.
- What are the common failure modes in prompting, and how do you debug them?
  - Answer: Common failures include hallucination, ignored instructions, wrong format, excessive verbosity, over-refusal, prompt injection, bias, ambiguity, and sensitivity to wording. Debug by isolating the failing input, simplifying the prompt, removing conflicting instructions, adding examples or schemas, checking retrieved context, lowering randomness, validating outputs, and testing against a known evaluation set.
- How do you handle edge cases and adversarial inputs in prompt design?
  - Answer: Identify likely edge cases up front, including empty input, malformed input, long input, ambiguous requests, unsupported languages, toxic content, and instruction attacks. Add explicit behavior for each class, use delimiters around untrusted content, validate inputs and outputs, limit tool permissions, and include adversarial examples in evaluation.
- What is the "lost in the middle" problem in long-context prompting?
  - Answer: Lost in the middle is the tendency for models to use information near the beginning or end of a long context more reliably than information buried in the middle. To reduce it, retrieve only the most relevant context, put critical instructions and evidence in salient positions, use summaries or section indexes, reorder passages by relevance, and ask for citations. Reference: [The Lost in the Middle Problem in LLMs](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms)
- What are output parsers, and why are they needed for production applications?
  - Answer: Output parsers convert model responses into validated application data, such as JSON objects, enums, database rows, or function arguments. They are needed because natural-language output can be malformed, incomplete, or inconsistent. Parsers enable schema validation, retries, error handling, security checks, and reliable integration with downstream code.
- How do you handle multi-language prompting effectively?
  - Answer: Detect or accept the target language explicitly, write instructions in the same language when possible, and use examples that match the target locale. Avoid idioms that do not translate well, preserve named entities carefully, handle script direction and formatting, and evaluate with native or high-quality multilingual test data. For critical tasks, combine multilingual retrieval, translation checks, and human review.
- Your few-shot prompting gives inconsistent results across similar inputs. How do you stabilize it?
  - Answer: Use more representative examples, order them consistently, remove ambiguous labels, and make the decision criteria explicit. Lower temperature, constrain output choices, use structured schemas, and add examples for edge cases that currently fail. If the task remains unstable, move to fine-tuning, prompt tuning, or a classifier-style model with a labeled evaluation set.
- Your LLM classification system is too sensitive to prompt wording changes. How do you reduce prompt sensitivity?
  - Answer: Define labels precisely, include positive and negative examples, require a fixed output enum, and calibrate on a validation set. Use lower temperature, shorter prompts, and stable templates. For production classification, consider embedding-based classifiers, fine-tuning, majority voting across prompt variants, or confidence thresholds with human review for uncertain cases.
- Your chatbot's system prompt containing proprietary business logic is being leaked by users. How do you prevent it?
  - Answer: Do not rely on secrecy of the prompt as the only protection. Move proprietary logic into backend code, tools, policies, or retrieval systems that expose only necessary results. Add instructions not to reveal hidden prompts, filter attempts to extract them, limit what the model can access, avoid placing secrets in prompts, and monitor for leakage attempts.
- Your LLM agent is vulnerable to prompt injection that reveals the system prompt. How do you defend it?
  - Answer: Treat all user and retrieved content as untrusted data. Keep system instructions and secrets outside tool-visible context when possible, use least-privilege tools, require confirmation for sensitive actions, validate tool arguments, and add policy checks before revealing internal information. The agent should ignore instructions inside documents that ask it to change roles, reveal prompts, or bypass rules. Reference: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- Your chain-of-thought prompting is not improving LLM accuracy on reasoning tasks. What do you fix?
  - Answer: First check whether the task actually benefits from explicit reasoning. Improve problem decomposition, add high-quality worked examples, reduce irrelevant context, and require verification of the final answer. Try self-consistency, tool use, retrieval, or a stronger reasoning model. Also evaluate whether exposed chain-of-thought is producing plausible but wrong reasoning; in production, prefer concise rationale plus answer checks.
- Your AI system works in English but fails for other languages. How do you add multilingual support?
  - Answer: Add language detection, localized prompts, multilingual examples, and test sets for each supported language. Use multilingual embeddings or retrieval indexes, preserve locale-specific formats, and evaluate with native speakers or trusted benchmarks. For lower-resource languages, consider translation pipelines, domain glossaries, and fallback flows when confidence is low.
- Your zero-shot cross-lingual transfer from English fails on other languages. How do you fix it?
  - Answer: Add target-language examples, translate and localize instructions, use multilingual or language-specific models, and evaluate each language separately. Improve retrieval with multilingual embeddings and language-aware metadata. For important tasks, collect labeled data in the target languages and fine-tune or calibrate the system instead of assuming English prompts will transfer.

### Retrieval-Augmented Generation (RAG)

- What is Retrieval-Augmented Generation (RAG), and why is it important?
  - Answer: RAG is a pattern where an application retrieves relevant external knowledge and passes it to an LLM before generation. It is important because it grounds answers in current or private data, reduces hallucination, improves citation support, and avoids retraining the model every time knowledge changes.
- Explain the architecture of a basic RAG system.
  - Answer: A basic RAG system has an offline indexing path and an online query path. Offline, documents are loaded, cleaned, chunked, embedded, and stored in a vector index with metadata. Online, a user query is embedded, relevant chunks are retrieved and optionally reranked, then the LLM generates an answer using the retrieved context and returns citations.
- What are the key components of a RAG pipeline?
  - Answer: The core components are document ingestion, parsing, chunking, embedding generation, vector or search storage, query understanding, retrieval, filtering, reranking, prompt construction, answer generation, citation handling, evaluation, monitoring, and update/index management.
- What are chunking strategies, and how do you choose the right chunk size?
  - Answer: Chunking strategies decide how documents are split before indexing. Choose chunk size based on document structure, embedding model limits, expected query granularity, and how much context the generator needs. Smaller chunks improve precise retrieval but may lose context; larger chunks preserve context but can add noise. Common approaches include fixed-size, recursive, semantic, sliding-window, and parent-child chunking. See [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag).
- Compare fixed-size chunking, semantic chunking, and recursive chunking.
  - Answer: Fixed-size chunking splits by token or character count and is simple but may cut across ideas. Semantic chunking groups text by meaning or topic boundaries, which improves coherence but costs more to compute. Recursive chunking splits by natural separators such as sections, paragraphs, sentences, and tokens, giving a practical balance between structure and predictable size. See [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag).
- What are embedding models, and how do they convert text to vectors?
  - Answer: Embedding models convert text into dense numeric vectors that capture semantic meaning. They tokenize the text, process it through a neural model, and output a fixed-size representation where similar meanings are close together under metrics such as cosine similarity or dot product. See [What are Embeddings?](https://outcomeschool.com/blog/what-are-embeddings).
- How do you choose an embedding model for your RAG system?
  - Answer: Choose based on retrieval quality for your domain, supported languages, context length, embedding dimension, latency, cost, deployment constraints, and compatibility with your vector database. Evaluate candidates on real queries with recall, precision, MRR, nDCG, and downstream answer quality rather than relying only on generic benchmarks.
- Explain Agentic RAG.
  - Answer: Agentic RAG lets an agent plan and control retrieval instead of doing a single fixed search. The agent can rewrite queries, call multiple retrievers or tools, inspect partial evidence, decide whether more retrieval is needed, and synthesize the final answer. It is useful for ambiguous, multi-step, or tool-heavy questions but adds latency and operational complexity. See [Agentic RAG](https://outcomeschool.com/blog/agentic-rag).
- What is hybrid search, and why is it better than pure vector search?
  - Answer: Hybrid search combines dense vector search with lexical search such as BM25. It is often better than pure vector search because vector search captures semantic similarity, while keyword search preserves exact matches for names, IDs, rare terms, error codes, and domain jargon. The combined score usually improves recall and robustness. See [How does Hybrid Search work?](https://outcomeschool.com/blog/how-does-hybrid-search-work).
- What is re-ranking, and how does it improve RAG retrieval quality?
  - Answer: Re-ranking takes an initial set of retrieved candidates and scores them with a more precise model, often a cross-encoder or LLM. It improves quality by moving the most relevant chunks to the top, removing weak matches, and giving the generator better evidence, at the cost of extra latency. See [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work).
- What is ColBERT, and how does late interaction retrieval work?
  - Answer: ColBERT is a retrieval architecture that stores token-level embeddings rather than only one vector per document. In late interaction, query and document token embeddings are computed separately, then matched at query time using efficient token-level similarity. This gives better precision than simple dense retrieval while remaining more scalable than full cross-encoder scoring. See [ColBERT - Late Interaction Retrieval Explained](https://outcomeschool.com/blog/decoding-colbert).
- Compare reranker architectures: cross-encoder, ColBERT, and LLM-based rerankers.
  - Answer: Cross-encoders jointly encode the query and candidate text, giving strong relevance judgments but higher latency. ColBERT uses late interaction over token embeddings, trading some scoring richness for better scalability. LLM-based rerankers can use reasoning, instructions, and domain criteria, but are usually the slowest and most expensive. In production, a common pattern is fast retrieval, then cross-encoder or ColBERT reranking, with LLM reranking only for high-value cases. See [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work) and [ColBERT - Late Interaction Retrieval Explained](https://outcomeschool.com/blog/decoding-colbert).
- How do you handle multi-document and multi-hop questions in RAG?
  - Answer: Use query decomposition, iterative retrieval, entity linking, graph traversal, and evidence aggregation. Retrieve evidence for each sub-question, track source provenance, then synthesize only claims supported by the collected evidence. Agentic RAG and GraphRAG are useful when the answer requires combining facts across documents. See [Agentic RAG](https://outcomeschool.com/blog/agentic-rag) and [GraphRAG](https://outcomeschool.com/blog/graphrag).
- What is the "lost in the middle" problem in RAG systems?
  - Answer: The lost in the middle problem is when an LLM pays less attention to information placed in the middle of a long context window than to information near the beginning or end. In RAG, this can make the model ignore relevant retrieved chunks. Mitigations include reranking, context compression, shorter prompts, placing the strongest evidence first or last, and splitting complex answers into smaller calls. See [The Lost in the Middle Problem in LLMs](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms).
- How do you evaluate a RAG system? Explain faithfulness, relevance, and context precision/recall.
  - Answer: Evaluate retrieval and generation separately and end-to-end. Faithfulness measures whether the answer is supported by the retrieved context. Relevance measures whether the answer addresses the user question. Context precision measures how much retrieved context is actually useful, while context recall measures whether the retriever found all evidence needed to answer. Use golden datasets, human review, LLM-as-judge with calibration, citation checks, and production feedback. See [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation).
- Explain Self-RAG. How does the model decide when to retrieve?
  - Answer: Self-RAG is a RAG approach where the model can decide whether retrieval is needed and critique whether retrieved evidence is useful. It may use learned control tokens, confidence thresholds, answerability checks, or reflection steps to choose between answering from existing context, retrieving more evidence, or abstaining. See [Agentic RAG](https://outcomeschool.com/blog/agentic-rag).
- What is GraphRAG, and when would you use it over traditional RAG?
  - Answer: GraphRAG builds or uses a graph of entities, relationships, communities, and documents, then retrieves over that graph plus source text. Use it when questions depend on relationships, multi-hop reasoning, global summaries, organizational knowledge, or connecting facts spread across many documents. Traditional vector RAG is usually simpler and better for direct semantic lookup. See [GraphRAG](https://outcomeschool.com/blog/graphrag).
- Vectorless RAG
  - Answer: Vectorless RAG retrieves context without dense vector embeddings. It can use keyword search, SQL queries, knowledge graphs, metadata filters, full-text search, or LLM/tool-driven lookup. It is useful when exact matching, structured data, compliance constraints, or simple infrastructure matter more than semantic vector similarity. See [Vectorless RAG](https://outcomeschool.com/blog/vectorless-rag).
- How do you handle structured data (tables, SQL databases) in a RAG pipeline?
  - Answer: Keep structured data in structured systems when possible. Use text-to-SQL, query builders, schema-aware tools, table parsers, and metadata-aware retrieval instead of flattening everything into plain text. For tables in documents, preserve headers, row/column relationships, units, and captions; for databases, generate safe queries, validate them, execute with least privilege, and pass compact results to the LLM.
- What are the common failure modes of RAG systems, and how do you debug them?
  - Answer: Common failures include bad parsing, poor chunking, weak embeddings, missing metadata, low recall, irrelevant retrieval, duplicate chunks, stale documents, access-control leaks, prompt misuse, and unsupported generation. Debug by logging each stage, inspecting retrieved chunks for test queries, measuring retrieval metrics, running answer faithfulness checks, adding negative tests, and tracing whether the issue is ingestion, retrieval, reranking, prompting, or generation.
- How do you handle document updates and maintain freshness in a RAG system?
  - Answer: Track document IDs, versions, timestamps, checksums, and source-of-truth locations. Use incremental indexing, upserts, deletions, tombstones, recrawling schedules, and freshness metadata. At query time, filter or boost by recency when appropriate, and make sure stale chunks are removed from both vector and keyword indexes.
- How do you optimize RAG for latency in production?
  - Answer: Optimize each stage: cache embeddings and frequent queries, use approximate nearest neighbor indexes, tune top-k, prefilter with metadata, parallelize retrieval sources, use fast rerankers only on a small candidate set, compress context, stream generation, and choose lower-latency models when quality allows. Monitor p50, p95, and p99 latency by pipeline stage.
- What is the role of metadata filtering in RAG systems?
  - Answer: Metadata filtering restricts retrieval to documents that satisfy attributes such as user permissions, tenant, date, product, language, document type, region, or version. It improves relevance, freshness, security, and latency by reducing the candidate search space before or during vector search. See [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work).
- Compare RAG vs fine-tuning. When would you use each?
  - Answer: Use RAG when the model needs access to external, private, fast-changing, or citeable knowledge. Use fine-tuning when you need to change model behavior, style, format, domain task performance, or tool-use patterns. They are complementary: a fine-tuned model can still use RAG for knowledge grounding.
- Long context windows keep getting cheaper. When should you use retrieval (RAG) vs putting everything in the context window?
  - Answer: Put everything in context when the corpus is small, the task requires global reasoning, and latency/cost are acceptable. Use RAG when the corpus is large, frequently updated, permissioned, or when you need precise citations and lower prompt cost. Long context does not remove the need for retrieval because irrelevant context can still distract the model and worsen lost-in-the-middle behavior. See [The Lost in the Middle Problem in LLMs](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms).
- What is query transformation in RAG (HyDE, query decomposition, step-back prompting)?
  - Answer: Query transformation rewrites or expands a user query before retrieval. HyDE generates a hypothetical answer and embeds it to retrieve similar documents. Query decomposition breaks a complex question into sub-queries. Step-back prompting creates a broader conceptual query before retrieving details. These methods improve recall for ambiguous or multi-hop questions. See [How does HyDE work in RAG?](https://outcomeschool.com/blog/how-does-hyde-work).
- How do you implement citation and source attribution in RAG?
  - Answer: Store source metadata with every chunk, including document ID, title, URL/path, page, section, offsets, and version. During generation, pass chunk IDs with the context and require the model to cite only those IDs. Validate citations by checking that each cited source supports the claim, and avoid citing sources that were not retrieved.
- How do you scale a RAG system to millions of documents?
  - Answer: Use distributed ingestion, incremental indexing, sharded vector and keyword indexes, approximate nearest neighbor search, metadata partitioning, compression or quantization, caching, and observability around recall and latency. Keep document lifecycle management reliable so updates and deletes propagate correctly. See [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work).
- What is parent-child chunking, and how does it improve retrieval?
  - Answer: Parent-child chunking indexes smaller child chunks for precise retrieval but returns a larger parent section for generation. This improves recall and precision because the search can match focused snippets while the LLM receives enough surrounding context to answer correctly. See [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag).
- Your RAG system is hallucinating despite having the right context. How do you fix it?
  - Answer: First confirm the right context is actually included in the final prompt and not buried in noise. Then tighten instructions to answer only from context, require citations, lower temperature, improve context ordering, remove conflicting chunks, use answerability checks, and add a faithfulness validator or retry path. If needed, split reasoning into extract-then-generate so the model quotes or selects supporting facts before composing the answer.
- Your RAG chunk overlap causes redundant results. How do you reduce redundancy?
  - Answer: Reduce overlap size, tune chunk boundaries, deduplicate near-identical chunks by hash or embedding similarity, group results by parent document, and apply maximal marginal relevance or diversity-aware reranking. You can also retrieve more candidates internally but pass only the most diverse supporting chunks to the generator.
- Your RAG retrieval is too slow with a large knowledge base. How do you speed it up?
  - Answer: Use ANN indexes, sharding, metadata prefilters, smaller candidate pools, embedding caching, query/result caching, optimized vector dimensions, quantization, and asynchronous parallel retrieval. Rerank only a small top-k set and monitor which stage dominates latency. See [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work).
- Your RAG system returns duplicate results. How do you deduplicate?
  - Answer: Deduplicate during ingestion with document checksums and canonical IDs, and during retrieval with exact text hashes, normalized titles/URLs, parent document grouping, and embedding-similarity clustering. In the final context, collapse duplicates and keep the highest-quality or most recent source.
- Your RAG system needs per-user access control on internal documents. How do you implement it?
  - Answer: Enforce permissions before generation, preferably during retrieval with tenant, group, role, and document ACL metadata filters. Sync ACLs from the source system, use least-privilege service accounts, test for cross-user leakage, log access decisions, and recheck permissions when documents or user roles change. Never rely only on the LLM prompt to enforce access control.
- Your RAG system fails on domain-specific jargon. How do you fix it?
  - Answer: Add domain examples to evaluation, use a domain-appropriate embedding model, add synonyms and acronym expansion, improve chunking around glossary-like content, and use hybrid search so exact jargon matches are not lost. For highly specialized domains, consider continued pretraining, fine-tuning embeddings, or adding a curated terminology layer. See [How does Hybrid Search work?](https://outcomeschool.com/blog/how-does-hybrid-search-work).
- Your text-only RAG system now needs to handle images and tables. How do you extend it?
  - Answer: Add multimodal ingestion. For images, create captions, OCR text, object/diagram descriptions, and image embeddings when visual similarity matters. For tables, preserve structure with HTML/Markdown tables or structured records rather than flattening blindly. Store modality metadata and route queries to text, table, image, or multimodal retrievers as needed.
- Your RAG knowledge base gets updated frequently and needs versioning. How do you manage it?
  - Answer: Store immutable document versions with stable document IDs, version IDs, timestamps, checksums, and source metadata. Use incremental indexing with upserts and tombstones, keep index snapshots for rollback, and expose version filters for queries. For answers, cite the version used so results can be audited later.
- Your RAG system fails on multi-hop questions that require combining multiple facts. How do you fix it?
  - Answer: Add query decomposition, iterative retrieval, entity-aware search, graph retrieval, and an evidence aggregation step. Retrieve for each sub-question, verify each fact against its source, then synthesize the final answer with citations for every major claim. See [Agentic RAG](https://outcomeschool.com/blog/agentic-rag) and [GraphRAG](https://outcomeschool.com/blog/graphrag).
- Your enterprise RAG system returns contradictory answers from different source documents. How do you resolve conflicts?
  - Answer: Preserve source metadata such as timestamp, owner, document type, authority level, and version. Rank authoritative and recent sources higher, detect conflicting claims, and ask the model to surface the conflict instead of hiding it when both sources are credible. For critical workflows, define business rules or human review paths for conflict resolution.
- Your RAG system returns outdated answers from an evolving knowledge base. How do you keep it current?
  - Answer: Implement freshness controls across ingestion and retrieval: source change detection, scheduled recrawls, incremental reindexing, delete propagation, recency metadata, and freshness-aware ranking. Monitor stale-answer reports and compare cited document versions against the current source of truth.
- Your RAG system struggles with PDF documents containing tables and layouts. How do you fix PDF parsing?
  - Answer: Use layout-aware PDF parsing, OCR for scanned pages, table extraction tools, and document structure preservation. Keep page numbers, headings, reading order, captions, and table coordinates as metadata. Validate parsed output with samples, route complex tables to table-specific extraction, and avoid indexing broken text streams that destroy row/column meaning.

### AI Agents and Agentic Systems

- What is an AI agent, and how does it differ from a simple LLM call?
  - Answer: An AI agent is a system that uses a model to pursue a goal through a loop of reasoning, acting, observing results, and deciding what to do next. A simple LLM call usually takes one input and returns one output. An agent adds state, tools, memory, planning, stop conditions, and often the ability to interact with external systems such as APIs, files, browsers, databases, or code environments.
- AI Agent Memory
  - Answer: AI agent memory is the information an agent preserves and reuses across steps or sessions. It can include the current conversation, task state, user preferences, retrieved documents, previous actions, tool outputs, and learned facts. Good memory design decides what to store, when to retrieve it, how to summarize it, how long to keep it, and how to avoid stale or unsafe information.
- Harness Engineering in AI
  - Answer: Harness engineering is the design of the software system around the model: prompts, tools, context assembly, memory, retries, validation, permissions, evals, logging, and UX. In agentic systems, the harness often matters as much as the model because it determines what the model can see, what it can do, how errors are handled, and how safely the system behaves.
- What matters more for an agentic coding tool like Claude Code: the model or the harness?
  - Answer: Both matter, but the harness is what turns a strong model into a useful coding agent. The model provides reasoning, code understanding, and generation quality. The harness provides repository search, file editing, terminal execution, test feedback, context management, permissions, checkpoints, and recovery. A weaker harness can waste a strong model; a strong harness can make model behavior far more reliable.
- Explain the ReAct (Reasoning + Acting) agent architecture.
  - Answer: ReAct is an agent pattern where the model alternates between reasoning and actions. It thinks about the next step, chooses an action such as calling a tool, observes the result, updates its reasoning, and repeats until it can answer or finish the task. This helps agents solve tasks that require external information, multi-step reasoning, or interaction with an environment.
- What is the Plan-and-Execute agent pattern?
  - Answer: Plan-and-Execute separates planning from execution. A planner first decomposes a goal into steps, then an executor performs each step, often using tools and updating the plan when reality differs from expectations. It is useful for complex tasks because it makes progress more structured, but it can be brittle if the initial plan is too rigid.
- What is tool use (function calling) in LLMs, and how does it enable agents?
  - Answer: Tool use, or function calling, lets an LLM request a predefined operation using structured arguments. The application executes the function and returns the result to the model. This enables agents because the model can go beyond text generation and interact with external systems: searching, writing files, querying databases, sending messages, running code, or triggering workflows.
- What is the difference between structured output and function calling?
  - Answer: Structured output makes the model respond in a required schema, such as JSON for extraction or classification. Function calling makes the model select a callable tool and provide arguments for it. Structured output is mainly about formatting the answer; function calling is about asking the host application to perform an action.
- How do you design and define tools for an AI agent?
  - Answer: Design tools around clear, narrow capabilities with explicit names, descriptions, input schemas, output schemas, permissions, and failure modes. A good tool exposes the right abstraction, not raw internal complexity. Include examples, validate arguments, return concise machine-readable results, and separate safe read-only tools from tools that mutate state or perform risky actions.
- How does an agent decide when to call a tool versus answering from its own knowledge?
  - Answer: The agent should call a tool when it needs fresh data, private data, exact computation, external side effects, verification, or access to information outside the model context. It can answer directly when the question is general, stable, already supported by context, and does not require action. Tool descriptions, system instructions, uncertainty thresholds, and eval feedback all shape this decision.
- What is the difference between single-agent and multi-agent systems?
  - Answer: A single-agent system uses one agent loop to plan, act, and respond. A multi-agent system uses multiple specialized agents that may collaborate, critique, delegate, or compete. Multi-agent systems can improve specialization and parallelism, but they add coordination cost, latency, and more places for mistakes to compound.
- When do multi-agent systems break down, and when is a single agent the better choice?
  - Answer: Multi-agent systems break down when roles are unclear, agents duplicate work, communication is noisy, decisions lack ownership, or coordination costs exceed the benefit. A single agent is often better for linear tasks, low-latency workflows, small context, simple tool use, or when one coherent decision-maker reduces ambiguity.
- What is Model Context Protocol (MCP), and how does it standardize tool integration?
  - Answer: Model Context Protocol, or MCP, is a protocol for connecting AI applications to external tools, resources, and prompts through a standard client-server interface. Instead of every app building custom integrations, an MCP server can expose capabilities in a consistent way, and an MCP client can discover and use them.
- How does MCP differ from traditional function calling?
  - Answer: Traditional function calling is usually defined inside one application: the developer gives the model a set of functions and executes selected calls. MCP standardizes how external servers expose tools and resources to AI clients. Function calling is the model-to-app action mechanism; MCP is an integration protocol for discovering and connecting external capabilities.
- What are AI SubAgents?
  - Answer: AI subagents are specialized agents invoked by a main agent to handle a narrower task. For example, a coding assistant might use one subagent for test analysis, another for security review, and another for documentation. Subagents help isolate context, specialize prompts and tools, and parallelize work, but they require clear delegation and result synthesis.
- What are the different types of agent memory (short-term, long-term, episodic)?
  - Answer: Short-term memory is the current working context, including the active conversation, plan, and recent tool results. Long-term memory stores durable facts, preferences, summaries, or knowledge across sessions. Episodic memory stores records of past events or task histories, such as what the agent tried, what worked, and what failed.
- How do you handle agent failures and implement error recovery?
  - Answer: Handle failures with explicit error types, retries with limits, fallback tools, validation, checkpoints, human escalation, and good observability. The agent should distinguish between transient failures, invalid inputs, unavailable tools, permission errors, and impossible tasks. Recovery often means summarizing state, revising the plan, trying a safer alternative, or asking for human approval.
- What is an agent loop, and how does it decide when to stop?
  - Answer: An agent loop repeatedly observes state, reasons about the next step, acts, receives feedback, and updates state. It stops when the goal is achieved, a final answer is produced, a tool or policy says to stop, the agent reaches a step/time/cost limit, progress stalls, confidence is too low, or human input is required.
- Context Engineering
  - Answer: Context engineering is the practice of selecting, structuring, compressing, and ordering the information given to a model. It includes system instructions, user request, retrieved knowledge, memory, tool schemas, examples, intermediate state, and constraints. The goal is to give the model enough relevant information to act well without wasting tokens or introducing distracting context.
- How does context compaction work?
  - Answer: Context compaction reduces a long interaction or large state into a smaller representation that preserves the important facts, decisions, goals, constraints, and open tasks. It may use summarization, extraction, pruning, hierarchical memory, or checkpoint files. Good compaction keeps task-critical details and drops redundant logs, stale reasoning, and irrelevant text.
- Loop Engineering
  - Answer: Loop engineering is the design of the control loop that drives an agent: when it reasons, when it calls tools, how it observes results, how it retries, when it asks for help, and when it stops. It focuses on making multi-step behavior reliable, bounded, observable, and recoverable.
- Graph Engineering
  - Answer: Graph engineering models an agent workflow as nodes and edges instead of one open-ended loop. Nodes represent steps such as classify, retrieve, call tool, validate, escalate, or respond. Edges define transitions based on state or outcomes. This makes complex workflows easier to reason about, test, resume, and constrain.
- How AI Agents Communicate?
  - Answer: Agents communicate through structured messages, shared state, task handoffs, tool outputs, events, queues, or a coordinator. Good communication includes clear roles, compact summaries, explicit requests, confidence levels, citations or evidence, and machine-readable outputs. Without structure, agents can amplify ambiguity and waste context.
- What are Agent Skills?
  - Answer: Agent skills are reusable capability packages that teach an agent how to perform a class of tasks. A skill may include instructions, examples, scripts, templates, tools, constraints, and validation steps. Skills help agents behave consistently on repeated workflows without putting every detail into the base prompt.
- How do you evaluate and test AI agents?
  - Answer: Evaluate agents with task-level success metrics, tool-call accuracy, trajectory analysis, safety checks, latency, cost, and human review. Tests should include golden tasks, adversarial cases, regression suites, simulated tools, real integration tests, and production monitoring. For agents, it is not enough to grade only the final answer; you also inspect the steps taken.
- What are the security risks of agentic systems, and how do you mitigate them?
  - Answer: Key risks include prompt injection, data exfiltration, unsafe tool use, privilege escalation, supply-chain attacks, insecure code execution, and accidental destructive actions. Mitigate them with least-privilege tools, sandboxing, allowlists, confirmation gates, input/output filtering, secret isolation, audit logs, policy checks, and separating untrusted content from trusted instructions.
- Your agent reads untrusted content (emails, web pages, documents) and can call tools. How do you prevent indirect prompt injection and data exfiltration?
  - Answer: Treat external content as data, never as instructions. Separate trusted system instructions from untrusted text, restrict tool permissions, block secret access, require user confirmation for sensitive actions, and use allowlists for destinations and operations. Add detectors for suspicious instructions, redact sensitive data from context, and log tool calls for audit.
- What is the difference between reactive and proactive agents?
  - Answer: Reactive agents respond only when triggered by a user request or event. Proactive agents monitor state, detect opportunities or risks, and initiate actions or suggestions without a direct prompt. Proactive agents need stronger guardrails because they can act at unexpected times and may affect users or systems without immediate supervision.
- How do you manage token consumption and cost in long-running agent workflows?
  - Answer: Manage cost by limiting loop steps, compacting context, retrieving only relevant memory, caching tool results, using smaller models for simple steps, summarizing long outputs, pruning unused tools, batching calls, and setting explicit token and dollar budgets. Track cost per task and stop or escalate when the expected value no longer justifies more work.
- What is the human-in-the-loop pattern for agents, and when is it needed?
  - Answer: Human-in-the-loop means the agent pauses for human review, approval, correction, or decision-making. It is needed for irreversible actions, high-cost operations, legal or financial impact, safety-sensitive domains, ambiguous goals, low confidence, policy exceptions, or when the agent needs authority it should not have automatically.
- How do you implement guardrails for AI agents to prevent harmful actions?
  - Answer: Implement guardrails at multiple layers: prompt policy, tool permissions, schema validation, action allowlists, sandboxing, output moderation, confirmation gates, rate limits, and monitoring. Do not rely only on the model to self-police. Risky actions should require deterministic checks and, when appropriate, human approval.
- What is agent reflection, and how does it improve agent performance?
  - Answer: Agent reflection is a step where the agent reviews its own output, plan, or tool trajectory to identify mistakes and improve the next attempt. It can catch missing requirements, bad assumptions, failed tool calls, or weak reasoning. Reflection improves quality when bounded and evidence-based, but it can waste tokens if used after every trivial step.
- What is the difference between code-generating agents and tool-calling agents?
  - Answer: Code-generating agents produce or modify code as their main output, often using tests and repository tools for feedback. Tool-calling agents primarily invoke predefined external functions to complete tasks. Many real systems combine both: a coding agent may call tools, and a tool-calling agent may generate small scripts or queries.
- How do you handle multi-modal inputs and outputs in agentic systems?
  - Answer: Use modality-specific parsers and models to convert images, audio, video, documents, or UI screens into structured representations the agent can reason over. Preserve references to original media when exact details matter. For outputs, route generation to the right renderer or tool, validate format and accessibility, and keep text, visual, and action state synchronized.
- How do you implement state management in complex agent workflows?
  - Answer: Use an explicit state object that records task goal, current step, plan, messages, tool results, decisions, errors, budgets, and user approvals. Store checkpoints so workflows can resume after failure. In graph-based frameworks, each node reads and updates state through typed transitions, which makes behavior more testable than implicit prompt-only state.
- How do you build a customer support agent with escalation logic?
  - Answer: Build it around intent detection, knowledge retrieval, policy checks, customer context, action tools, and escalation triggers. The agent should answer routine questions, perform safe account actions, and escalate when confidence is low, the user is upset, policy requires human review, identity verification fails, or the request involves refunds, legal issues, or high-value accounts.
- What is agent orchestration, and how do you implement it?
  - Answer: Agent orchestration is the coordination of models, tools, agents, state, and workflows to complete a task. It can be implemented with a central controller, a workflow graph, a planner-executor pattern, queues, event-driven services, or a multi-agent supervisor. The orchestrator manages routing, dependencies, retries, budgets, permissions, and final synthesis.
- What is Sakana Fugu, and how does it orchestrate a team of AI models?
  - Answer: Sakana Fugu is an agentic system from Sakana AI that treats multiple AI models as a coordinated team. Instead of relying on one model for every step, it can assign roles to different models, combine their outputs, and use orchestration to improve problem solving. The key idea is model collaboration: use specialized strengths and aggregation rather than a single monolithic response.
- How do you build a code execution agent safely using sandboxed environments?
  - Answer: Run code in isolated sandboxes with restricted filesystem, network, CPU, memory, runtime, and secrets. Use disposable environments, dependency allowlists, timeouts, resource limits, malware scanning when needed, and clear separation between generated code and production systems. Log executions, require approvals for risky operations, and never expose credentials inside the sandbox unless absolutely necessary.
- Your AI agent is stuck in an infinite loop. How do you detect and break the cycle?
  - Answer: Detect loops with max-step limits, repeated action signatures, unchanged state, repeated tool calls, low progress metrics, and similar reasoning summaries across iterations. Break the cycle by forcing a plan revision, changing tools, compacting state, asking for human input, returning a partial result, or stopping with a clear failure reason.
- Your AI agent gets conflicting answers from different tools. How does it reconcile them?
  - Answer: Reconcile conflicts by ranking sources by authority, freshness, precision, and relevance. Ask whether the tools answer the same question, compare timestamps and units, rerun or cross-check critical calls, and expose uncertainty when needed. For high-stakes decisions, the agent should escalate or require human review rather than averaging incompatible answers.
- Your AI agent burns too many tokens per task. How do you reduce token consumption?
  - Answer: Reduce tokens by shortening prompts, pruning tool lists, summarizing long histories, retrieving smaller chunks, caching repeated results, using structured outputs, limiting chain-of-thought style verbosity, and using cheaper models for routing or extraction. Also cap loop iterations and avoid sending full raw tool outputs when a concise parsed result is enough.
- Your AI agent keeps exceeding its budget per task. How do you enforce budget limits?
  - Answer: Give each task a budget for steps, tokens, tool calls, wall-clock time, and money. Track usage centrally before and after every model or tool call. When the budget is near exhaustion, force summarization, switch to cheaper strategies, return a partial answer, or ask for approval to continue. Budget checks should be enforced by the runtime, not only by prompts.
- Your AI agent hallucinates tool capabilities and passes wrong inputs. How do you fix it?
  - Answer: Improve tool schemas, names, descriptions, examples, and validation errors. Limit the visible tool set to relevant tools, use strict JSON schema validation, return clear error messages, and add few-shot examples for correct calls. For critical tools, add a pre-execution validator or planner step that checks whether the selected tool and parameters match the user goal.
- Your AI agent deleted a production database. How do you prevent irreversible actions?
  - Answer: Use least-privilege credentials, separate read and write tools, block destructive operations by default, require explicit human approval, add environment protections, use backups, dry runs, soft deletes, change windows, and policy checks. The agent should never have direct unrestricted production admin access for irreversible operations.
- Your AI agent has many tools, but keeps picking the wrong one. How do you improve tool selection?
  - Answer: Reduce the number of visible tools, group tools by task, improve descriptions, add examples, create a routing step, and log tool-selection failures for evals. Tool names should be specific and action-oriented. If tools overlap, merge them or make their differences explicit. Retrieval-based tool selection can expose only the most relevant tools per step.
- Your AI agent takes too long to complete a task. How do you speed it up?
  - Answer: Profile where time is spent: model calls, tool latency, serial execution, retries, or context size. Speed it up with parallel tool calls, smaller models for simple steps, caching, fewer loop iterations, better planning, shorter context, streaming partial results, and direct deterministic code for steps that do not need an LLM.
- Your long-running agent drifts after hours and confidently works on the wrong thing. How do you diagnose and fix it?
  - Answer: Diagnose drift by reviewing checkpoints, state summaries, tool trajectories, goal changes, memory retrievals, and compaction outputs. Fix it with explicit goal state, periodic revalidation against the original objective, user-approved milestones, stronger stop conditions, better memory hygiene, and checkpoints that allow rollback to the last correct state.
- Your LLM selects the right tool but extracts the wrong parameters. How do you fix parameter extraction?
  - Answer: Use stricter schemas, field descriptions, examples, validation, and clarification questions for missing or ambiguous values. Add a parameter verification step before execution, normalize units and dates, and return specific validation errors to the model. For important workflows, use a separate extraction model or deterministic parser plus tests.
- How do Computer-Use Agents work?
  - Answer: Computer-use agents operate a graphical interface by observing screenshots, accessibility trees, or DOM state, then taking actions such as clicking, typing, scrolling, or using keyboard shortcuts. They run an observe-plan-act loop over a live environment. They need strong safeguards because UI state can be ambiguous and actions may affect real accounts or systems.
- How does LangChain work?
  - Answer: LangChain is a framework for building LLM applications by composing models, prompts, retrievers, tools, memory, and chains. It provides abstractions for common patterns such as RAG, tool calling, agents, document loading, and integrations. It is useful for prototyping and orchestration, though production systems still need careful state, eval, and observability design.
- How does LangGraph work?
  - Answer: LangGraph models agent workflows as stateful graphs. Each node performs a step, edges determine what happens next, and shared state carries information through the workflow. It supports cycles, checkpoints, persistence, human review, and resumability, making it well suited for complex agents that need more control than a simple chain.
- What is OKF (Open Knowledge Format)?
  - Answer: OKF, or Open Knowledge Format, is a proposed structured way to represent knowledge so AI systems can ingest, retrieve, exchange, and reason over it more reliably than with unstructured text alone. The idea is to make knowledge portable and machine-readable through explicit entities, relationships, metadata, and provenance.

### Fine-Tuning and Model Adaptation

- What is fine-tuning, and when should you fine-tune an LLM?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- Explain the difference between full fine-tuning and parameter-efficient fine-tuning (PEFT).
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- What is LoRA (Low-Rank Adaptation), and how does it work?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- What is QLoRA, and how does it enable fine-tuning on consumer hardware?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- How does fine-tuning work?
  - Answer: [How does fine-tuning work?](https://outcomeschool.com/blog/how-does-fine-tuning-work)
- Explain Prefix Tuning and Prompt Tuning. How are they different from LoRA?
  - Answer: [How does Prefix Tuning work?](https://outcomeschool.com/blog/how-does-prefix-tuning-work)
- What is adapter-based fine-tuning?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- Explain the difference between pre-training, supervised fine-tuning (SFT), and preference optimization (RLHF/DPO).
  - Answer: [Decoding InstructGPT](https://outcomeschool.com/blog/decoding-instructgpt)
- What is RLHF (Reinforcement Learning from Human Feedback), and how is it used to align LLMs?
  - Answer: [Reinforcement Learning from Human Feedback (RLHF)](https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf)
- What is Deep RL from Human Preferences, the paper that started RLHF?
  - Answer: [Deep RL from Human Preferences](https://outcomeschool.com/blog/decoding-deep-rl-from-human-preferences)
- Why did DPO displace PPO-based RLHF at many labs? When is online RL still better?
  - Answer: [Direct Preference Optimization (DPO)](https://outcomeschool.com/blog/direct-preference-optimization-dpo) and [Proximal Policy Optimization (PPO)](https://outcomeschool.com/blog/proximal-policy-optimization-ppo)
- What is instruction tuning, and why is it important for chat models?
  - Answer: [Decoding InstructGPT](https://outcomeschool.com/blog/decoding-instructgpt)
- How do you prepare a dataset for fine-tuning an LLM?
- What is catastrophic forgetting, and how do you prevent it during fine-tuning?
  - Answer: [Continual Learning in LLMs](https://outcomeschool.com/blog/continual-learning-in-llms)
- When should you choose fine-tuning over RAG over prompt engineering?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- How do you evaluate a fine-tuned model's performance?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- What is synthetic data generation, and how do you use it for fine-tuning?
- What are the key hyperparameters for fine-tuning (learning rate, epochs, batch size, LoRA rank)?
  - Answer: [LoRA - Low-Rank Adaptation of LLMs](https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms)
- Do the GPU memory math for full fine-tuning a 7B model in bf16 with Adam. Now with LoRA and QLoRA.
- How do you fine-tune a model for a specific domain (legal, medical, finance)?
- What is continual pre-training, and when would you use it?
- How do you merge multiple LoRA adapters?
  - Answer: [LoRA - Low-Rank Adaptation of LLMs](https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms)
- What is the difference between SFT (Supervised Fine-Tuning) and alignment training?
  - Answer: [Decoding InstructGPT](https://outcomeschool.com/blog/decoding-instructgpt)
- What is RLAIF (RL from AI Feedback), and how does it differ from RLHF?
- What is Constitutional AI, and how does it differ from RLHF?
- What is RLVR (Reinforcement Learning with Verifiable Rewards), and when does it beat a learned reward model?
  - Answer: [Group Relative Policy Optimization (GRPO)](https://outcomeschool.com/blog/group-relative-policy-optimization-grpo)
- What is knowledge distillation for fine-tuning, and what are the legal considerations?
  - Answer: [How does Knowledge Distillation work?](https://outcomeschool.com/blog/how-does-knowledge-distillation-work)
- Your fine-tuned LLM produces factually wrong outputs due to training data quality issues. How do you fix it?
- You must choose between LoRA and full fine-tuning for a domain-specific assistant. How do you decide?
  - Answer: [LoRA - Low-Rank Adaptation of LLMs](https://outcomeschool.com/blog/lora-low-rank-adaptation-of-llms)
- Your fine-tuned model memorized training data verbatim instead of learning patterns. How do you fix overfitting?
- Your fine-tuned LLM forgot its general capabilities after domain-specific fine-tuning. How do you fix catastrophic forgetting?
  - Answer: [Continual Learning in LLMs](https://outcomeschool.com/blog/continual-learning-in-llms)
- Your RLHF preference data has low annotator agreement. How do you ensure data quality?

### Vector Databases and Embeddings

- What are embeddings in the context of AI engineering?
  - Answer: [Embeddings in Machine Learning](https://www.youtube.com/watch?v=LedXW6xl21s)
- How do embedding models convert text to vectors?
  - Answer: [What are Embeddings?](https://outcomeschool.com/blog/what-are-embeddings)
- What is Contrastive Learning, and how is it used to train embedding models?
  - Answer: [What is Contrastive Learning?](https://outcomeschool.com/blog/contrastive-learning)
- What is the difference between sparse and dense embeddings?
- Explain cosine similarity, dot product, and Euclidean distance for vector search.
  - Answer: [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- What is a vector database, and how does it differ from a traditional database?
  - Answer: [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- How does Approximate Nearest Neighbor (ANN) search work?
  - Answer: [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)
- Compare HNSW, IVF, and flat indexes. How do you pick one, and what does recall@k cost in latency?
  - Answer: [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)
- How does an Embedding Cache work?
  - Answer: [How does an Embedding Cache work?](https://outcomeschool.com/blog/how-does-an-embedding-cache-work)
- How do you choose the right embedding model for your use case?
- What is embedding dimensionality, and how does it affect performance and cost?
- How do you handle embedding drift when the embedding model is updated?
- What are multi-modal embeddings, and how are they generated?
  - Answer: [Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)
- How do you index and query multi-tenant data in a vector database?
- What is quantization of embeddings, and how does it reduce storage costs?
- How do you benchmark and evaluate embedding model quality?
- What is the role of metadata in vector databases?
  - Answer: [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- How do you handle large-scale vector search with billions of vectors?
  - Answer: [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)
- What is hybrid search (combining keyword search with vector search)?
  - Answer: [How does Hybrid Search work?](https://outcomeschool.com/blog/how-does-hybrid-search-work)
- How do you fine-tune an embedding model for a specific domain?
  - Answer: [What is Contrastive Learning?](https://outcomeschool.com/blog/contrastive-learning)
- Your vector database for RAG is consuming too much memory. How do you reduce it?
- Your vector database cannot scale to millions of embeddings. How do you fix the bottleneck?
  - Answer: [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)
- Your new embedding model has different dimensions from the existing vectors in production. How do you handle the mismatch?
- Your vector search returns irrelevant results despite high similarity scores. How do you fix it?
- You deployed a new embedding model, and search quality crashed overnight. How do you handle embedding drift?
- Your semantic search fails for short queries. How do you improve it?

### AI System Design

- Design a Real-Time Voice AI Agent
  - Answer: [Design a Real-Time Voice AI Agent](https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent)
- Design ChatGPT: Training to Serving (End to End)
- Design a RAG System (Chat with Your Documents)
- Design an enterprise RAG assistant over 10M documents with per-user permissions.
- Design Memory for a Personal AI Assistant
  - Answer: [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)
- Design a Deep Research Agent
- Design a Multi-Agent Customer Support System
  - Answer: [Multi-Agent Systems](https://outcomeschool.com/blog/multi-agent-systems)
- Design an On-Device AI Assistant
  - Answer: [Cloud vs On-Device Model Deployment](https://outcomeschool.com/blog/cloud-vs-on-device-model-deployment) and [How does llama.cpp run LLMs on everyday hardware?](https://outcomeschool.com/blog/how-does-llama-cpp-run-llms-on-everyday-hardware)
- Design a Multimodal Search System (Text, Image, Video)
- Design an LLM Inference Platform (vLLM-as-a-Service)
  - Answer: [How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work) and [LLM Inference Optimization](https://outcomeschool.com/blog/llm-inference-optimization)
- Design an LLM Evaluation Platform
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- Design a Text-to-Image Generation Service (Midjourney-like)
- Design a Music Generation Service (Suno-like)
- Design a Video Generation Service (Sora-like)
- Design an AI Coding Agent.
  - Answer: [How does Claude Code work?](https://outcomeschool.com/blog/how-does-claude-code-work) and [How does Cursor work?](https://outcomeschool.com/blog/how-does-cursor-work)
- Design a code generation and review system.
- Design a content moderation system using AI.
- Design a real-time AI recommendation system.
- Design an AI-powered email assistant.
- Design a medical diagnosis assistant using AI.
- Design a fraud detection system powered by LLMs.
- Design an AI-powered data extraction pipeline from unstructured documents.
- Design a Text-to-SQL system over a data warehouse with thousands of tables.
- Design a personalized learning assistant.
- Design an AI system for automated code migration.
- Design an AI-powered legal document review system.
- Design a conversational AI system with memory across sessions.
  - Answer: [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)
- How do you design for latency vs quality trade-offs in AI systems?
- How do you implement caching strategies for LLM applications?
  - Answer: [How does Prompt Caching work?](https://outcomeschool.com/blog/how-does-prompt-caching-work) and [How does Semantic Caching work?](https://outcomeschool.com/blog/how-does-semantic-caching-work)
- How do you design rate limiting and cost management for AI APIs?
- How do you handle failover and fallback strategies for AI systems?
- How do you design an AI system for high availability and fault tolerance?
- How do you design an AI system that gracefully degrades when the model is unavailable?
- What are the key considerations for multi-region deployment of AI systems?
- Design an AI-powered search engine for an e-commerce platform.
- Design an AI gateway/proxy for managing LLM access across an organization.
- How do you design a RAG system that handles conflicting information across sources?
- How do you approach capacity planning for an AI system?
- Design a multi-tenant AI chatbot platform where each business gets a custom chatbot.
- Design an AI meeting summarizer system for thousands of meetings daily.
- Design an AI notification system that prioritizes instead of broadcasting.
- Design an AI-powered anomaly detection system for cloud infrastructure.
- Design an AI-powered document processing pipeline for financial institutions.
- Design an AI dynamic pricing engine.
- Design an AI resume screening system that handles 100K applications per week.
- Design an AI voice assistant architecture.
  - Answer: [Design a Real-Time Voice AI Agent](https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent)
- Design a multi-agent workflow system where agents collaborate on complex tasks.
  - Answer: [Multi-Agent Systems](https://outcomeschool.com/blog/multi-agent-systems)
- Design a real-time AI transcription system for concurrent audio streams.
- Design an AI-powered live streaming content moderation system.

### LLMOps and Production AI

- How does Prompt Caching work?
  - Answer: [How does Prompt Caching work?](https://outcomeschool.com/blog/how-does-prompt-caching-work)
- Prefill vs Decode
  - Answer: [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)
- Why is prefill compute-bound and decode memory-bandwidth-bound?
  - Answer: [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)
- What is chunked prefill, and why does it improve tail latency under mixed traffic?
  - Answer: [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)
- What is Prefill-Decode Disaggregation, and when does it pay off?
  - Answer: [Prefill-Decode Disaggregation in LLM Inference](https://outcomeschool.com/blog/prefill-decode-disaggregation)
- Explain the AI product lifecycle from ideation to production.
- What is LLMOps, and how does it differ from traditional MLOps?
- How do you serve LLMs in production?
  - Answer: [How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work) and [LLM Inference Optimization](https://outcomeschool.com/blog/llm-inference-optimization)
- What is model quantization?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk) and [How does Model Quantization work?](https://outcomeschool.com/blog/how-does-model-quantization-work)
- Explain post-training quantization (PTQ) vs quantization-aware training (QAT). What breaks when you push weights to 2-4 bits?
  - Answer: [How does Model Quantization work?](https://outcomeschool.com/blog/how-does-model-quantization-work)
- How do you monitor LLM applications in production?
  - Answer: [AI Agent Observability](https://outcomeschool.com/blog/ai-agent-observability)
- What is LLM observability?
  - Answer: [AI Agent Observability](https://outcomeschool.com/blog/ai-agent-observability)
- What are guardrails for LLMs, and how do you implement them?
  - Answer: [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- How do you implement content filtering for AI outputs?
  - Answer: [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- How do you estimate the cost of running an AI-powered feature in production?
- How do you optimize LLM inference costs in production?
  - Answer: [LLM Inference Optimization](https://outcomeschool.com/blog/llm-inference-optimization)
- How do you implement A/B testing for LLM systems?
- What is CI/CD for AI applications, and how does it differ from traditional CI/CD?
- How do you version and manage prompts in production?
- What is model versioning, and how do you handle model rollbacks?
- How do you implement rate limiting and throttling for LLM APIs?
- How do you handle model updates and migrations without downtime?
- What is the role of feature flags in AI deployments?
- How do you implement logging and tracing for LLM applications?
  - Answer: [AI Agent Observability](https://outcomeschool.com/blog/ai-agent-observability)
- How do you handle PII and sensitive data in LLM inputs and outputs?
- What is a gateway pattern for LLM API management?
- How does Token Streaming work?
  - Answer: [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)
- How do you implement streaming responses for real-time AI applications?
  - Answer: [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)
- How does vLLM work?
  - Answer: [How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work)
- How does SGLang work?
  - Answer: [How does SGLang work?](https://outcomeschool.com/blog/how-does-sglang-work)
- How does TensorRT-LLM work?
  - Answer: [How does TensorRT-LLM work?](https://outcomeschool.com/blog/how-does-tensorrt-llm-work)
- How does llama.cpp run LLMs on everyday hardware?
  - Answer: [How does llama.cpp run LLMs on everyday hardware?](https://outcomeschool.com/blog/how-does-llama-cpp-run-llms-on-everyday-hardware)
- When would you choose vLLM vs SGLang vs TensorRT-LLM?
  - Answer: [How does vLLM work?](https://outcomeschool.com/blog/how-does-vllm-work), [How does SGLang work?](https://outcomeschool.com/blog/how-does-sglang-work) and [How does TensorRT-LLM work?](https://outcomeschool.com/blog/how-does-tensorrt-llm-work)
- What are the key SLAs and metrics for production AI systems (latency, throughput, availability)?
- Cloud vs on-device Model Deployment for AI applications.
  - Answer: [Cloud vs On-Device Model Deployment](https://outcomeschool.com/blog/cloud-vs-on-device-model-deployment)
- How do you implement fallback strategies when the primary model is unavailable or rate-limited?
- How do you implement structured output from LLMs reliably in production?
  - Answer: [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- How do you handle long contexts efficiently in production (context compression, prefix caching)?
  - Answer: [How does Prompt Caching work?](https://outcomeschool.com/blog/how-does-prompt-caching-work) and [How does context compaction work?](https://outcomeschool.com/blog/how-does-context-compaction-work)
- What is semantic routing, and how do you implement it in a multi-model system?
  - Answer: [LLM Routing](https://outcomeschool.com/blog/llm-routing)
- How do you manage secrets and API keys securely in LLM applications?
- Your LLM API has latency spikes during peak hours. How do you stabilize it?
- Your LLM endpoint's p99 latency doubled after a deploy with no model change. How do you diagnose it?
- Your LLM costs are too high in production. How do you reduce costs without degrading quality?
  - Answer: [LLM Routing](https://outcomeschool.com/blog/llm-routing) and [How does Semantic Caching work?](https://outcomeschool.com/blog/how-does-semantic-caching-work)
- Your application is hitting LLM provider rate limits during peak hours. How do you handle it?
- Your application depends on one LLM provider. How do you switch providers without downtime?
- Your AI system handles 100 requests/sec but crashes at 5000. How do you scale for concurrent requests?
- A traffic spike brings down your AI system. How do you handle peak traffic?
- One LLM provider outage took down your entire system. How do you eliminate single points of failure?
- Your multi-LLM pipeline fails when one model in the chain breaks. How do you handle orchestration failure?
  - Answer: [AI Orchestration](https://outcomeschool.com/blog/ai-orchestration)
- Your AI pipeline has zero visibility into which step is failing. How do you add observability?
  - Answer: [AI Agent Observability](https://outcomeschool.com/blog/ai-agent-observability)
- You quantized your LLM, but accuracy dropped significantly. How do you minimize quantization loss?
  - Answer: [How does Model Quantization work?](https://outcomeschool.com/blog/how-does-model-quantization-work)
- One failing AI component can take down your entire platform. How do you design graceful degradation?

### Evaluation and Testing

- AI Agent Evaluation
  - Answer: [AI Agent Evaluation](https://outcomeschool.com/blog/ai-agent-evaluation)
- LLM Evaluation
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- AI Agent Observability
  - Answer: [AI Agent Observability](https://outcomeschool.com/blog/ai-agent-observability)
- What is evaluation-driven development for AI applications?
- Why is AI only as good as our definition of done?
  - Answer: [AI Is Only as Good as Our Definition of Done](https://outcomeschool.com/blog/ai-is-only-as-good-as-our-definition-of-done)
- How do you evaluate LLM outputs? What metrics do you use?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- Explain BLEU, ROUGE, and BERTScore. When would you use each?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- What is G-Eval, and how does it use LLMs for evaluation?
  - Answer: [LLM as a Judge](https://outcomeschool.com/blog/llm-as-a-judge)
- What is LLM-as-a-judge evaluation, and what are its limitations?
  - Answer: [LLM as a Judge](https://outcomeschool.com/blog/llm-as-a-judge)
- How do you conduct human evaluation for AI systems?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- What is red teaming, and how do you red team an LLM application?
- How do you detect and measure hallucinations in LLM outputs?
- What is adversarial testing for AI systems?
- How do you build a regression test suite for AI applications?
- What are benchmark suites (MMLU, HumanEval, GSM8K), and how do you interpret them?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- What is benchmark contamination, and how do you guard against it?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- Your new model version scores higher on every benchmark, but users say it got worse. Why does this happen, and how do you find the problem?
- How do you evaluate a RAG system end-to-end?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- How do you evaluate the quality of AI agents?
  - Answer: [AI Agent Evaluation](https://outcomeschool.com/blog/ai-agent-evaluation)
- How would you evaluate an autonomous coding agent? Why can SWE-bench pass rates be misleading?
  - Answer: [AI Agent Evaluation](https://outcomeschool.com/blog/ai-agent-evaluation)
- What is the difference between offline and online evaluation for AI systems?
- How do you measure factual consistency in LLM outputs?
- How do you evaluate multi-turn conversation quality?
- What is the role of golden datasets in AI evaluation?
- How do you build an eval set when there is no labelled ground truth and domain experts are expensive?
- How do you implement continuous evaluation for production AI systems?
- How do you evaluate bias in AI model outputs?
- How do you compare two models or prompts in a statistically rigorous way?
- How do you evaluate the robustness of an LLM application across input variations?
- What are the key differences between evaluating traditional ML vs LLM applications?
- How do you set up an evaluation framework from scratch for a new LLM application?
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- Your model passes one fairness metric but fails another. How do you handle conflicting audit results?
- Your model was fair at deployment, but became biased 6 months later. How do you monitor continuously?
- An external auditor cannot reproduce your model's results. How do you ensure audit reproducibility?
- How do you structure red teaming for an LLM chatbot before launch?
- How do you red team a multimodal model where text-only safety tests miss cross-modal attacks?

### AI Safety, Ethics, and Responsible AI

- What are hallucinations in LLMs, and how do you mitigate them?
- What is prompt injection, and what are the different types (direct, indirect)?
  - Answer: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- How do you implement input and output guardrails for AI systems?
  - Answer: [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- What is AI alignment, and why is it important?
  - Answer: [Decoding InstructGPT](https://outcomeschool.com/blog/decoding-instructgpt) and [Reinforcement Learning from Human Feedback (RLHF)](https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf)
- How do you detect and mitigate bias in AI systems?
- What are the key data privacy considerations (GDPR, CCPA) when building AI applications?
- How do you handle PII in LLM inputs and outputs?
- What is explainability in AI, and why does it matter?
- What is the difference between interpretability and explainability?
- What is mechanistic interpretability, and why do AI labs invest in it?
- How do you build trust with users in AI-powered applications?
- What are adversarial attacks on AI systems, and how do you defend against them?
- What is data poisoning, and how can it affect AI models?
- How do you implement content safety filters for AI-generated content?
  - Answer: [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- What is responsible AI, and what frameworks exist for implementing it?
- How do you handle copyright and intellectual property concerns with AI-generated content?
- What is the EU AI Act, and how does it affect AI engineering?
- How do you implement audit trails and logging for AI decisions?
- What is model card documentation, and why is it important?
- How do you handle misuse and abuse of AI systems in production?
- What is differential privacy, and how can it be applied during model training?
- How would you design an AI incident response plan?
- What is the NIST AI Risk Management Framework (AI RMF)?
- Your healthcare chatbot gives medical diagnoses it should not make. How do you add safety guardrails?
  - Answer: [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- Your AI system is reproducing copyrighted material verbatim. How do you prevent this?
- Your resume screening AI rejects more female candidates for engineering roles. How do you fix gender bias?
- Your AI model passes bias checks by gender and race separately, but fails for intersectional groups. How do you handle it?
- Your AI denied a loan, and the customer demands a GDPR explanation. How do you provide one?
- A user invokes the right to be forgotten, but their data is in your model weights. How do you comply?
- The EU AI Act may classify your AI system as high-risk. How do you comply?
- Your differentially private model lost significant accuracy. How do you balance privacy and utility?
- One malicious participant is poisoning your federated learning model. How do you defend against it?
- Your AI hiring model uses proxy features for protected attributes. How do you eliminate proxy discrimination?
- Your predictive model creates a feedback loop of biased outcomes. How do you break it?
- Your AI generates fake news images. How do you implement watermarking for AI-generated content?
  - Answer: [How Does LLM Watermarking Work?](https://outcomeschool.com/blog/how-does-llm-watermarking-work)
- Your AI denies a service, and the user has no way to challenge it. How do you design an appeals process?
- An auditor asks why your AI rejected a request 6 months ago, and you have no logs. How do you build audit trails?
- You removed PII, but users were re-identified from anonymized data. How do you prevent re-identification?
- A pre-trained model from an open-source repo may contain a hidden backdoor. How do you detect it?
- Your LLM's training data was deliberately poisoned by an adversary. How do you respond?
- Your AI mental health chatbot gave harmful advice to a user in crisis. How do you mitigate harm?
- Your AI system caused incorrect critical decisions. How do you run a blameless post-mortem?
- Radiologists agree with AI 98% of the time, even when it is wrong. How do you prevent human over-reliance on AI?
- Your content moderation flags normal cultural expressions as offensive in other markets. How do you adapt cross-culturally?
- Your AI training produces massive carbon emissions. How do you reduce environmental impact?

### Multimodal AI

- What are Multimodal AI models, and how do they process different types of data?
  - Answer: [Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)
- How do vision-language models process images?
  - Answer: [Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)
- How do Image Embeddings work?
  - Answer: [How do Image Embeddings work?](https://outcomeschool.com/blog/how-do-image-embeddings-work)
- How does CLIP work, and why is it important for multi-modal AI?
  - Answer: [What is Contrastive Learning?](https://outcomeschool.com/blog/contrastive-learning) and [How do Image Embeddings work?](https://outcomeschool.com/blog/how-do-image-embeddings-work)
- What are the key architectures for multi-modal models?
  - Answer: [Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)
- How does image generation work with diffusion models (Stable Diffusion, DALL-E, Flux)?
  - Answer: [Diffusion Models](https://outcomeschool.com/blog/diffusion-models)
- What is text-to-speech (TTS), and what models are used for it?
- How does speech-to-text (Whisper) work?
- Budget the end-to-end latency for a real-time voice agent (VAD, ASR, LLM, TTS, network). Where does the time go?
  - Answer: [Design a Real-Time Voice AI Agent](https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent)
- How do you handle barge-in (user interruptions) in a voice agent?
  - Answer: [Design a Real-Time Voice AI Agent](https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent)
- Cascaded ASR + LLM + TTS vs native speech-to-speech models: what are the trade-offs?
  - Answer: [Design a Real-Time Voice AI Agent](https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent)
- What is multi-modal RAG, and how does it differ from text-only RAG?
- How do you build a system that processes both images and text?
  - Answer: [Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)
- What are multi-modal embeddings, and how are they used for cross-modal search?
  - Answer: [Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)
- How do you evaluate multi-modal AI systems?
- What are the challenges of real-time multi-modal AI processing?
- How do you handle video understanding with AI?
- What is visual question answering (VQA)?
- What is document understanding, and how do models parse documents with layouts?
- How do you fine-tune a vision-language model?
- What are the latency and cost considerations for multi-modal AI in production?
- How do you handle multi-modal content moderation?
- What is text-to-video generation, and what are the current state-of-the-art approaches?
- Explain Multimodal Fusion Techniques: Early Fusion vs Late Fusion.
- Your vision-language model generates factually incorrect image descriptions. How do you fix it?
- Your VLM answers single-image questions but fails on multi-page documents. How do you fix it?
- Your multimodal LLM ignores the image and generates descriptions from text alone. How do you fix it?
- Your diffusion model ignores precise control requirements in text prompts. How do you improve controllability?
- Your diffusion model generates sharp but repetitive images. How do you balance quality vs diversity?
- Your diffusion model takes too long per image. How do you speed up sampling?

### AI Infrastructure and Scalability

- How do you improve inference speed in production LLM deployments?
  - Answer: [LLM Inference Optimization](https://www.youtube.com/watch?v=jV2sCj4lHYk)
- LLM optimization techniques
  - Answer: [LLM optimization techniques](https://www.linkedin.com/posts/pallavi-shekhar_5-llm-optimization-techniques-lets-understand-activity-7442067281532325888-4aOS)
- How do you select GPUs for LLM inference?
  - Answer: [How does a GPU work for Deep Learning?](https://outcomeschool.com/blog/how-does-a-gpu-work-for-deep-learning)
- How does a GPU work for Deep Learning?
  - Answer: [How does a GPU work for Deep Learning?](https://outcomeschool.com/blog/how-does-a-gpu-work-for-deep-learning)
- How does a Google TPU work?
  - Answer: [How does a Google TPU work?](https://outcomeschool.com/blog/how-does-a-google-tpu-work)
- How does an LPU work?
  - Answer: [How does an LPU work?](https://outcomeschool.com/blog/how-does-an-lpu-work)
- Estimate the GPU memory needed to serve a 70B model (weights, KV cache, activations). Does it fit on a single 80 GB GPU?
- Do the roofline math: how many tokens/sec can one H100 produce for a 70B model at batch size 1?
- What is model parallelism vs data parallelism in distributed training?
- What is tensor parallelism, and how does it help serve large models?
- What is pipeline parallelism?
- How does continuous batching improve LLM inference throughput?
  - Answer: [Continuous Batching in LLMs](https://outcomeschool.com/blog/continuous-batching-in-llms)
- What is speculative decoding, and how does it speed up inference?
  - Answer: [Speculative Decoding](https://outcomeschool.com/blog/speculative-decoding)
- How does Medusa (multi-head speculative decoding) work?
  - Answer: [Medusa - Multi-Head Speculative Decoding](https://outcomeschool.com/blog/decoding-medusa)
- How does EAGLE (feature-level speculative decoding) work?
  - Answer: [EAGLE - Feature-Level Speculative Decoding](https://outcomeschool.com/blog/decoding-eagle)
- What is N-gram Speculation in LLMs, and how does it speed up generation?
  - Answer: [N-gram Speculation in LLMs](https://outcomeschool.com/blog/n-gram-speculation-in-llms)
- What is KV cache, and how do you manage memory for it?
  - Answer: [What is KV Cache in LLMs?](https://outcomeschool.com/blog/kv-cache-in-llms)
- What is Paged Attention?
  - Answer: [Paged Attention in LLMs](https://outcomeschool.com/blog/paged-attention-in-llms)
- How does GGUF work?
  - Answer: [How does GGUF work?](https://outcomeschool.com/blog/how-does-gguf-work)
- How do you optimize inference for edge and mobile deployment?
  - Answer: [How does llama.cpp run LLMs on everyday hardware?](https://outcomeschool.com/blog/how-does-llama-cpp-run-llms-on-everyday-hardware) and [How does Model Quantization work?](https://outcomeschool.com/blog/how-does-model-quantization-work)
- What is model quantization (INT8, INT4, FP16, BF16), and how does it affect quality?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk) and [How does Model Quantization work?](https://outcomeschool.com/blog/how-does-model-quantization-work)
- How do you implement auto-scaling for AI workloads?
- What is the role of load balancing in AI serving infrastructure?
- How do you manage GPU memory for serving multiple models?
- What is model sharding, and when would you use it?
- How do you implement request queuing and priority scheduling for AI services?
- What are the cost trade-offs between self-hosted and API-based AI inference?
- How do you handle cold start latency for serverless AI deployments?
- How do you implement model caching to reduce redundant computations?
  - Answer: [How does Prompt Caching work?](https://outcomeschool.com/blog/how-does-prompt-caching-work)
- What is the difference between synchronous and asynchronous inference, and when do you use each?
- What is FSDP (Fully Sharded Data Parallel), and how does it differ from DeepSpeed ZeRO?
- What are TTFT, TPOT, and throughput, and how do they trade against each other?
  - Answer: [Prefill vs Decode: LLM Inference Optimization](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)
- How do you monitor and profile LLM inference in production (TTFT, inter-token latency, GPU utilization)?
- What is model routing at the infrastructure level, and how do you route requests based on complexity and cost?
  - Answer: [LLM Routing](https://outcomeschool.com/blog/llm-routing)

### Coding and Practical Implementation

- Implement a basic RAG pipeline using an embedding model and a vector database.
- Build a simple AI agent with tool use (e.g., calculator, web search).
  - Answer: [ReAct Agent](https://outcomeschool.com/blog/react-agent)
- Implement semantic search using embeddings and cosine similarity.
  - Answer: [How does Semantic Search work?](https://outcomeschool.com/blog/how-does-semantic-search-work) and [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- Write code for different text chunking strategies (fixed-size, recursive, semantic).
  - Answer: [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)
- Implement a prompt template system with variable substitution.
- Build an evaluation pipeline for LLM outputs using LLM-as-a-judge.
  - Answer: [LLM as a Judge](https://outcomeschool.com/blog/llm-as-a-judge)
- Implement streaming responses for an LLM API.
  - Answer: [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)
- Build a simple vector similarity search from scratch.
  - Answer: [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- Implement a conversation memory system for a chatbot (sliding window, summary, buffer).
  - Answer: [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)
- Write code to detect and handle hallucinations in LLM outputs.
- Implement a retry mechanism with exponential backoff for LLM API calls.
- Write a function calling (tool use) handler for an LLM API.
  - Answer: [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- Implement a simple re-ranker for search results.
  - Answer: [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work)
- Build a basic document parser that extracts text from PDFs and splits it into chunks.
  - Answer: [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)
- Implement cosine similarity, dot product, and Euclidean distance functions from scratch.
  - Answer: [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- Write code to implement token counting and context window management.
- Build a simple prompt versioning system.
- Implement a caching layer for LLM responses.
- Implement semantic caching for LLM queries (cache responses for semantically similar queries).
  - Answer: [How does Semantic Caching work?](https://outcomeschool.com/blog/how-does-semantic-caching-work)
- Write code to detect prompt injection attempts in user inputs.
  - Answer: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- Implement an LLM output guardrails system that checks for off-topic responses and PII leakage.
  - Answer: [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- Build a multi-agent system where agents have different roles and collaborate on a task.
  - Answer: [Multi-Agent Systems](https://outcomeschool.com/blog/multi-agent-systems)
- Implement scaled dot-product attention with a causal mask from scratch (NumPy or PyTorch).
  - Answer: [Math behind Attention - Q, K, and V](https://outcomeschool.com/blog/math-behind-attention-qkv) and [Causal Masking in Attention](https://outcomeschool.com/blog/causal-masking-in-attention)
- Implement multi-head attention, then convert it to grouped-query attention.
  - Answer: [Multi-Head Attention in Transformers](https://outcomeschool.com/blog/multi-head-attention-in-transformers) and [Grouped Query Attention](https://outcomeschool.com/blog/grouped-query-attention)
- Implement a KV cache and single-step decode for causal multi-head attention.
  - Answer: [What is KV Cache in LLMs?](https://outcomeschool.com/blog/kv-cache-in-llms)
- Implement BPE (Byte Pair Encoding) training and encoding from scratch.
  - Answer: [Byte Pair Encoding](https://outcomeschool.com/blog/bpe-in-llms)
- Implement top-k, top-p, and temperature sampling over a logits vector.
- Implement an LRU cache with O(1) get/put, then add per-entry TTL.
- Implement a token-bucket rate limiter for an LLM API where cost scales with tokens, then make it distributed.
- Write an async batch processor that runs an LLM call over 50,000 documents with a concurrency limit, retries with jitter, and error isolation.
- Write a streaming SSE parser for LLM token streams that handles arbitrary chunk boundaries.
  - Answer: [How does Token Streaming work?](https://outcomeschool.com/blog/how-does-token-streaming-work)
- Implement a minimal agent loop with tool dispatch, error handling, and a step budget.
  - Answer: [AI Agent Loop](https://outcomeschool.com/blog/ai-agent-loop)

### Behavioral and Scenario-Based Questions

- What is AI Engineering, and how does it differ from Machine Learning Engineering?
- How do you decide whether a problem needs AI or a traditional software solution?
- How do you measure the ROI of an AI feature?
- How do you handle hallucinations when they occur in a production AI system?
- How do you decide between using an LLM API vs self-hosting an open-source model?
- How do you manage stakeholder expectations for AI projects?
- Describe your approach to debugging a poor-performing RAG system.
- How do you stay current with the rapidly evolving AI landscape?
- How do you balance innovation with reliability in AI systems?
- Tell me about a challenging AI project you worked on. What was the problem? What approach did you take? What trade-offs did you make? What was the outcome?
- How would you handle a situation where an AI model produces biased or harmful outputs in production?
- How do you approach cost optimization for an AI system that's exceeding budget?
- Describe a time when you had to choose between model accuracy and latency. How did you make the decision?
- How would you handle a situation where your AI system's quality degrades over time?
- How do you communicate AI limitations to non-technical stakeholders?
- How would you approach building an AI feature with limited labeled data?
- Describe your experience working with cross-functional teams on AI projects.
- Where do you see AI engineering heading in the next 3-5 years?
- Why are you interested in this AI engineering role?
- Your PM wants to ship an AI feature with a 15% hallucination rate on edge cases. How do you communicate the risk?
- A non-technical executive asks why your AI feature cannot be 100% accurate. How do you explain LLM limitations?
- You need to choose between a complex agentic system that scores 15% better on benchmarks, or a simpler RAG pipeline that is easier to maintain. How do you decide?

### License

```
   Copyright (C) 2026 Outcome School

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
