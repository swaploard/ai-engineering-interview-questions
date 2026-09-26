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
  - Answer: The Transformer is a neural architecture built around attention instead of recurrence. It represents tokens as embeddings, adds position information, and repeatedly applies attention, feed-forward layers, normalization, and residual connections. Attention lets each token weight other relevant tokens in the context, making Transformers highly parallelizable and effective for long-range language patterns.
- What are the key components of the Transformer architecture?
  - Answer: Key components include token embeddings, positional encodings or positional embeddings, attention layers, multi-head attention, feed-forward networks, residual connections, layer normalization, and an output projection to vocabulary logits. Encoder-decoder Transformers also include cross-attention between decoder tokens and encoder outputs.
- Walk me through what happens, step by step, in one forward pass of a decoder-only Transformer.
  - Answer: The input text is tokenized and converted to token embeddings. Position information is added, then each Transformer block applies masked self-attention so each token can attend only to allowed previous tokens. The result passes through residual connections, normalization, and a feed-forward network. After the final block, the model projects hidden states to vocabulary logits, and the last position's logits are used to choose the next token.
- What is tokenization in LLMs?
  - Answer: Tokenization is the process of converting text into smaller units, called tokens, that the model can process. Tokens may be words, subwords, characters, punctuation, or byte-level chunks. The tokenizer maps each token to an integer ID, which is then converted into an embedding. Tokenization affects cost, context usage, multilingual quality, and how well domain-specific terms are represented.
- Explain BPE (Byte Pair Encoding).
  - Answer: Byte Pair Encoding is a subword tokenization method that starts with small units, often bytes or characters, and repeatedly merges the most frequent adjacent pairs into larger tokens. This creates a vocabulary that can represent common words efficiently while still handling rare or unseen words by splitting them into smaller pieces.
- Explain WordPiece and SentencePiece.
  - Answer: WordPiece is a subword tokenizer that builds tokens by selecting pieces that improve the likelihood of the training corpus, commonly using continuation markers for subword fragments. SentencePiece is a tokenizer framework that treats text as a raw character stream and can train BPE or unigram models without relying on pre-tokenized whitespace. Both help models handle rare words, misspellings, and multilingual text with a fixed vocabulary.
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
  - Answer: A rough KV-cache estimate is batch_size * sequence_length * layers * 2 * kv_heads * head_dim * bytes_per_value. The factor of 2 is for keys and values. Memory grows linearly with batch size and context length, so longer prompts or more concurrent users reduce how many requests fit on a GPU. Models using MQA or GQA reduce kv_heads and therefore reduce cache memory.
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
- What is Cross-Entropy Loss?
  - Answer: Cross-entropy loss measures the difference between the model's predicted probability distribution and the true target distribution. In language modeling, the target is usually the correct next token, and the loss is lower when the model assigns that token high probability. Minimizing cross-entropy is equivalent to maximizing the likelihood of the training text.
- What is Grouped-Query Attention (GQA), and how does it differ from Multi-Head Attention (MHA)?
  - Answer: In standard multi-head attention, each query head has its own key and value heads. Grouped-Query Attention keeps many query heads but shares fewer key/value heads across groups of queries. This reduces KV-cache memory and decoding bandwidth while preserving more quality than using a single shared KV head as in Multi-Query Attention.
- How does Sliding Window Attention work?
  - Answer: Sliding Window Attention restricts each token to attend only to a fixed-size window of nearby tokens instead of the entire sequence. This lowers attention cost from quadratic over the full context to roughly linear in sequence length for a fixed window. It works well for local dependencies, but models may need special mechanisms such as global tokens, memory, retrieval, or attention sinks to preserve long-range information.
- How do Attention Sinks work?
  - Answer: Attention sinks are tokens, often early tokens in the sequence, that many later tokens attend to disproportionately. They help stabilize attention distributions during long-context generation because attention needs somewhere to place probability mass even when local tokens are not useful. Some long-context methods preserve these sink tokens while sliding or truncating the rest of the cache to maintain quality.
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
- Explain zero-shot, one-shot, and few-shot prompting with examples.
  - Answer: [Explain zero-shot, one-shot, and few-shot prompting with examples](https://www.linkedin.com/posts/pallavi-shekhar_llm-prompting-ai-activity-7441801012472078336-JsHr)
- What is chain-of-thought (CoT) prompting, and when should you use it?
  - Answer: [How does Chain-of-Thought (CoT) Prompting work?](https://outcomeschool.com/blog/how-does-chain-of-thought-prompting-work)
- Explain self-consistency prompting and how it improves reasoning.
- What is tree-of-thought prompting?
- What is ReAct (Reasoning + Acting) prompting, and how does it work?
  - Answer: [ReAct Agent](https://outcomeschool.com/blog/react-agent)
- What is a system prompt, and how does it influence model behavior?
- How do you structure prompts for consistent structured output (JSON, XML)?
- What is prompt injection, and how do you defend against it?
  - Answer: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- What is jailbreaking in LLMs, and what are common jailbreak techniques?
  - Answer: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- How do you optimize prompts for cost and latency?
- What is the difference between prompt engineering and prompt tuning?
- What is a prompt template, and how do you design one for production use?
- How do you handle multi-turn conversations with LLMs?
- What is role prompting, and when is it effective?
- What is prompt chaining, and how do you design a chain of prompts for complex tasks?
  - Answer: [How does Prompt Chaining work?](https://outcomeschool.com/blog/how-does-prompt-chaining-work)
- How do you evaluate and iterate on prompt quality?
- What are meta-prompts, and how can they be used to generate prompts?
- What are the common failure modes in prompting, and how do you debug them?
- How do you handle edge cases and adversarial inputs in prompt design?
- What is the "lost in the middle" problem in long-context prompting?
  - Answer: [The Lost in the Middle Problem in LLMs](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms)
- What are output parsers, and why are they needed for production applications?
- How do you handle multi-language prompting effectively?
- Your few-shot prompting gives inconsistent results across similar inputs. How do you stabilize it?
- Your LLM classification system is too sensitive to prompt wording changes. How do you reduce prompt sensitivity?
- Your chatbot's system prompt containing proprietary business logic is being leaked by users. How do you prevent it?
- Your LLM agent is vulnerable to prompt injection that reveals the system prompt. How do you defend it?
  - Answer: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- Your chain-of-thought prompting is not improving LLM accuracy on reasoning tasks. What do you fix?
- Your AI system works in English but fails for other languages. How do you add multilingual support?
- Your zero-shot cross-lingual transfer from English fails on other languages. How do you fix it?

### Retrieval-Augmented Generation (RAG)

- What is Retrieval-Augmented Generation (RAG), and why is it important?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- Explain the architecture of a basic RAG system.
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- What are the key components of a RAG pipeline?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- What are chunking strategies, and how do you choose the right chunk size?
  - Answer: [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)
- Compare fixed-size chunking, semantic chunking, and recursive chunking.
  - Answer: [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)
- What are embedding models, and how do they convert text to vectors?
  - Answer: [What are Embeddings?](https://outcomeschool.com/blog/what-are-embeddings)
- How do you choose an embedding model for your RAG system?
- Explain Agentic RAG.
  - Answer: [Agentic RAG](https://outcomeschool.com/blog/agentic-rag)
- What is hybrid search, and why is it better than pure vector search?
  - Answer: [How does Hybrid Search work?](https://outcomeschool.com/blog/how-does-hybrid-search-work)
- What is re-ranking, and how does it improve RAG retrieval quality?
  - Answer: [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work)
- What is ColBERT, and how does late interaction retrieval work?
  - Answer: [ColBERT - Late Interaction Retrieval Explained](https://outcomeschool.com/blog/decoding-colbert)
- Compare reranker architectures: cross-encoder, ColBERT, and LLM-based rerankers.
  - Answer: [How does a Reranker work?](https://outcomeschool.com/blog/how-does-a-reranker-work) and [ColBERT - Late Interaction Retrieval Explained](https://outcomeschool.com/blog/decoding-colbert)
- How do you handle multi-document and multi-hop questions in RAG?
  - Answer: [Agentic RAG](https://outcomeschool.com/blog/agentic-rag) and [GraphRAG](https://outcomeschool.com/blog/graphrag)
- What is the "lost in the middle" problem in RAG systems?
  - Answer: [The Lost in the Middle Problem in LLMs](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms)
- How do you evaluate a RAG system? Explain faithfulness, relevance, and context precision/recall.
  - Answer: [LLM Evaluation](https://outcomeschool.com/blog/llm-evaluation)
- Explain Self-RAG. How does the model decide when to retrieve?
  - Answer: [Agentic RAG](https://outcomeschool.com/blog/agentic-rag)
- What is GraphRAG, and when would you use it over traditional RAG?
  - Answer: [GraphRAG](https://outcomeschool.com/blog/graphrag)
- Vectorless RAG
  - Answer: [Vectorless RAG](https://outcomeschool.com/blog/vectorless-rag)
- How do you handle structured data (tables, SQL databases) in a RAG pipeline?
- What are the common failure modes of RAG systems, and how do you debug them?
- How do you handle document updates and maintain freshness in a RAG system?
- How do you optimize RAG for latency in production?
- What is the role of metadata filtering in RAG systems?
  - Answer: [How does a Vector Database work?](https://outcomeschool.com/blog/how-does-a-vector-database-work)
- Compare RAG vs fine-tuning. When would you use each?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- Long context windows keep getting cheaper. When should you use retrieval (RAG) vs putting everything in the context window?
  - Answer: [The Lost in the Middle Problem in LLMs](https://outcomeschool.com/blog/lost-in-the-middle-problem-in-llms)
- What is query transformation in RAG (HyDE, query decomposition, step-back prompting)?
  - Answer: [How does HyDE work in RAG?](https://outcomeschool.com/blog/how-does-hyde-work)
- How do you implement citation and source attribution in RAG?
- How do you scale a RAG system to millions of documents?
  - Answer: [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)
- What is parent-child chunking, and how does it improve retrieval?
  - Answer: [Chunking Strategies for RAG](https://outcomeschool.com/blog/chunking-strategies-for-rag)
- Your RAG system is hallucinating despite having the right context. How do you fix it?
- Your RAG chunk overlap causes redundant results. How do you reduce redundancy?
- Your RAG retrieval is too slow with a large knowledge base. How do you speed it up?
  - Answer: [How does Approximate Nearest Neighbor (ANN) search work?](https://outcomeschool.com/blog/how-does-approximate-nearest-neighbor-ann-search-work)
- Your RAG system returns duplicate results. How do you deduplicate?
- Your RAG system needs per-user access control on internal documents. How do you implement it?
- Your RAG system fails on domain-specific jargon. How do you fix it?
  - Answer: [How does Hybrid Search work?](https://outcomeschool.com/blog/how-does-hybrid-search-work)
- Your text-only RAG system now needs to handle images and tables. How do you extend it?
- Your RAG knowledge base gets updated frequently and needs versioning. How do you manage it?
- Your RAG system fails on multi-hop questions that require combining multiple facts. How do you fix it?
  - Answer: [Agentic RAG](https://outcomeschool.com/blog/agentic-rag) and [GraphRAG](https://outcomeschool.com/blog/graphrag)
- Your enterprise RAG system returns contradictory answers from different source documents. How do you resolve conflicts?
- Your RAG system returns outdated answers from an evolving knowledge base. How do you keep it current?
- Your RAG system struggles with PDF documents containing tables and layouts. How do you fix PDF parsing?

### AI Agents and Agentic Systems

- What is an AI agent, and how does it differ from a simple LLM call?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk) and [AI Agent Explained](https://outcomeschool.com/blog/ai-agent)
- AI Agent Memory
  - Answer: [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)
- Harness Engineering in AI
  - Answer: [Harness Engineering in AI](https://outcomeschool.com/blog/harness-engineering-in-ai)
- What matters more for an agentic coding tool like Claude Code: the model or the harness?
  - Answer: [Harness Engineering in AI](https://outcomeschool.com/blog/harness-engineering-in-ai) and [How does Claude Code work?](https://outcomeschool.com/blog/how-does-claude-code-work)
- Explain the ReAct (Reasoning + Acting) agent architecture.
  - Answer: [ReAct Agent](https://outcomeschool.com/blog/react-agent)
- What is the Plan-and-Execute agent pattern?
  - Answer: [Plan-and-Execute Agent](https://outcomeschool.com/blog/plan-and-execute-agent)
- What is tool use (function calling) in LLMs, and how does it enable agents?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk) and [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- What is the difference between structured output and function calling?
  - Answer: [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- How do you design and define tools for an AI agent?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk)
- How does an agent decide when to call a tool versus answering from its own knowledge?
- What is the difference between single-agent and multi-agent systems?
  - Answer: [Multi-Agent Systems](https://outcomeschool.com/blog/multi-agent-systems)
- When do multi-agent systems break down, and when is a single agent the better choice?
  - Answer: [Multi-Agent Systems](https://outcomeschool.com/blog/multi-agent-systems)
- What is Model Context Protocol (MCP), and how does it standardize tool integration?
  - Answer: Explained in this video: [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization](https://www.youtube.com/watch?v=lnfWvX66FUk) and [What is MCP (Model Context Protocol)?](https://outcomeschool.com/blog/what-is-mcp-model-context-protocol)
- How does MCP differ from traditional function calling?
  - Answer: [What is MCP (Model Context Protocol)?](https://outcomeschool.com/blog/what-is-mcp-model-context-protocol) and [How does Function Calling work in LLMs?](https://outcomeschool.com/blog/how-does-function-calling-work-in-llms)
- What are AI SubAgents?
  - Answer: [AI SubAgents](https://outcomeschool.com/blog/ai-subagents)
- What are the different types of agent memory (short-term, long-term, episodic)?
  - Answer: [AI Agent Memory](https://outcomeschool.com/blog/ai-agent-memory)
- How do you handle agent failures and implement error recovery?
- What is an agent loop, and how does it decide when to stop?
  - Answer: [AI Agent Loop](https://outcomeschool.com/blog/ai-agent-loop)
- Context Engineering
  - Answer: [Context Engineering](https://outcomeschool.com/blog/context-engineering)
- How does context compaction work?
  - Answer: [How does context compaction work?](https://outcomeschool.com/blog/how-does-context-compaction-work)
- Loop Engineering
  - Answer: [Loop Engineering](https://outcomeschool.com/blog/what-is-loop-engineering)
- Graph Engineering
  - Answer: [Graph Engineering](https://outcomeschool.com/blog/what-is-graph-engineering)
- How AI Agents Communicate?
  - Answer: [How AI Agents Communicate](https://outcomeschool.com/blog/how-ai-agents-communicate)
- What are Agent Skills?
  - Answer: [What are Agent Skills?](https://outcomeschool.com/blog/what-are-agent-skills)
- How do you evaluate and test AI agents?
  - Answer: [AI Agent Evaluation](https://outcomeschool.com/blog/ai-agent-evaluation)
- What are the security risks of agentic systems, and how do you mitigate them?
  - Answer: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- Your agent reads untrusted content (emails, web pages, documents) and can call tools. How do you prevent indirect prompt injection and data exfiltration?
  - Answer: [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)
- What is the difference between reactive and proactive agents?
- How do you manage token consumption and cost in long-running agent workflows?
  - Answer: [How does context compaction work?](https://outcomeschool.com/blog/how-does-context-compaction-work) and [How would you reduce the token consumption?](https://www.linkedin.com/posts/pallavi-shekhar_ai-aiagents-machinelearning-activity-7439550125015994368-LTmE)
- What is the human-in-the-loop pattern for agents, and when is it needed?
- How do you implement guardrails for AI agents to prevent harmful actions?
  - Answer: [How do LLM guardrails work?](https://outcomeschool.com/blog/how-do-llm-guardrails-work)
- What is agent reflection, and how does it improve agent performance?
  - Answer: [Reflection Agent](https://outcomeschool.com/blog/reflection-agent)
- What is the difference between code-generating agents and tool-calling agents?
- How do you handle multi-modal inputs and outputs in agentic systems?
- How do you implement state management in complex agent workflows?
  - Answer: [How does LangGraph work?](https://outcomeschool.com/blog/how-does-langgraph-work)
- How do you build a customer support agent with escalation logic?
- What is agent orchestration, and how do you implement it?
  - Answer: [AI Orchestration](https://outcomeschool.com/blog/ai-orchestration)
- What is Sakana Fugu, and how does it orchestrate a team of AI models?
  - Answer: [Sakana Fugu - The Technical Report Explained](https://outcomeschool.com/blog/decoding-sakana-fugu)
- How do you build a code execution agent safely using sandboxed environments?
- Your AI agent is stuck in an infinite loop. How do you detect and break the cycle?
  - Answer: [Fix an infinite loop in an AI agent](https://www.linkedin.com/posts/pallavi-shekhar_ai-aiagents-machinelearning-share-7440257380707364864-5Ycc)
- Your AI agent gets conflicting answers from different tools. How does it reconcile them?
- Your AI agent burns too many tokens per task. How do you reduce token consumption?
  - Answer: [How would you reduce the token consumption?](https://www.linkedin.com/posts/pallavi-shekhar_ai-aiagents-machinelearning-activity-7439550125015994368-LTmE)
- Your AI agent keeps exceeding its budget per task. How do you enforce budget limits?
  - Answer: [AI Agent Loop](https://outcomeschool.com/blog/ai-agent-loop)
- Your AI agent hallucinates tool capabilities and passes wrong inputs. How do you fix it?
- Your AI agent deleted a production database. How do you prevent irreversible actions?
- Your AI agent has many tools, but keeps picking the wrong one. How do you improve tool selection?
- Your AI agent takes too long to complete a task. How do you speed it up?
- Your long-running agent drifts after hours and confidently works on the wrong thing. How do you diagnose and fix it?
- Your LLM selects the right tool but extracts the wrong parameters. How do you fix parameter extraction?
- How do Computer-Use Agents work?
  - Answer: [How do Computer-Use Agents work?](https://outcomeschool.com/blog/how-do-computer-use-agents-work)
- How does LangChain work?
  - Answer: [How does LangChain work?](https://outcomeschool.com/blog/how-does-langchain-work)
- How does LangGraph work?
  - Answer: [How does LangGraph work?](https://outcomeschool.com/blog/how-does-langgraph-work)
- What is OKF (Open Knowledge Format)?
  - Answer: [What is OKF (Open Knowledge Format)?](https://outcomeschool.com/blog/what-is-okf-open-knowledge-format)

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
