# How This GPT Was Trained From Scratch: A Primer

*Pitched at roughly a "200-level" college course — assumes no machine learning or
programming background, but doesn't oversimplify. If you can follow a decent
explainer article, you can follow this.*

This document explains the ideas behind `nanogpt-scifi` for readers who want to
understand **how a language model actually learns**, without reading code. It's a
companion to two other documents in this repo, each aimed at a different reader:

| Document | Audience | What it covers |
|---|---|---|
| **`llm-primer.md`** (this doc) | Anyone curious how LLMs work, no background assumed | The concepts, explained from first principles |
| [`build-log.md`](build-log.md) | Engineers/reviewers interested in process | What was actually done, in what order, including bugs and dead ends |
| `pipeline-notes.md` *(forthcoming)* | Technical reviewers | A rigorous, cited technical report on this specific implementation and its results |

Each module below ends with a **"In this project"** box connecting the concept to a
real, specific choice or number from `nanogpt-scifi` — so the ideas aren't abstract.

---

## Module 0: Why Build One From Scratch?

Almost everyone's experience of "AI" today is typing into a chat box — using a
language model someone else already built and trained. That teaches you how to
*prompt* a model. It teaches you nothing about what happens *before* the chat box
exists: how raw text becomes something a model can learn from, what "training"
actually does to the model, or why the model behaves the way it does.

This project builds and trains a small language model from nothing — no pretrained
weights, no existing checkpoint — specifically to make that hidden process visible.

> **In this project:** the model starts as random noise (literally: its internal
> numbers are initialized randomly) and learns everything it knows about English
> sentence structure and sci-fi vocabulary purely from reading ~29 million characters
> of public-domain sci-fi novels, one small chunk at a time.

---

## Module 1: What Is a Language Model, Really?

Strip away the hype: a language model is a **next-token prediction machine**. Given
some text so far, it outputs a probability for every possible "what comes next" —
and the whole system is built by training something that gets progressively better
at guessing right.

"GPT" stands for **Generative Pre-trained Transformer**:
- **Generative** — it produces new text, one piece at a time, rather than just
  classifying or labeling existing text.
- **Pre-trained** — it learns general patterns from a large body of text before
  being used for anything specific (as opposed to being hand-programmed with rules).
- **Transformer** — the specific neural network architecture it uses (Module 4).

The model's "knowledge" is stored as **parameters** (also called weights) — plain
numbers, adjusted during training, that determine how it transforms an input into a
prediction. A model doesn't store sentences or facts the way a database does; it
stores numbers that, combined in the right sequence of mathematical operations,
tend to produce plausible next-tokens.

> **In this project:** the model has roughly 10 million parameters — tiny by modern
> standards (GPT-3 has 175 *billion*). That's a deliberate, laptop-scale choice
> (Module 7), not an attempt to compete with production models.

---

## Module 2: Turning Text Into Numbers (Tokenization)

Neural networks only operate on numbers, so the very first problem is: how do you
turn a sentence into numbers, and back?

The answer is **tokenization**: chop text into a fixed set of pieces (tokens), and
assign each distinct piece an integer ID. There are a few common granularities:

- **Character-level** — every individual character (`a`, `!`, a space) is its own
  token. Simple, small vocabulary, but the model has to work harder to learn that
  groups of characters form words.
- **Word-level** — every distinct word is a token. Intuitive, but the vocabulary
  balloons (every inflection, typo, and made-up word needs its own slot) and it
  can't handle words it's never seen.
- **Subword / BPE** (what production models like GPT-4 use) — a middle ground:
  common whole words get one token, rarer words get split into meaningful chunks
  (e.g. "unhappiness" → "un" + "happi" + "ness").

This project uses **character-level tokenization** — the simplest option, chosen
deliberately so that the tokenizer itself stays trivially understandable (a lookup
table, nothing more), keeping the learning focus on the transformer and training
loop rather than on subword-splitting algorithms.

The mechanics: scan the whole training corpus once, collect every distinct
character that appears, and assign each one an integer. This gives you two lookup
tables — one from character to number (`stoi`, "string to index") and its mirror
image, number back to character (`itos`, "index to string"). Encoding a sentence is
just looking up each character's number; decoding is the reverse.

> **In this project:** scanning the 28.9-million-character sci-fi corpus produced
> **251 distinct characters** — more than plain English (~65) because the corpus
> includes foreign-language titles and passages (accented letters, Greek letters,
> even runes) pulled in from the source texts. Every one of those 251 symbols got
> its own integer ID before any training began.

---

## Module 3: Turning Numbers Into Meaning (Embeddings)

A token's integer ID (say, "the letter `q` is token #47") is arbitrary — it carries
no information about what `q` *means* or how it relates to other characters. The
next step is to give the model room to learn that meaning for itself.

An **embedding** is a vector — a list of numbers, think of it as coordinates in a
space — assigned to each token. Instead of representing `q` as the bare number 47,
the model represents it as a point in (in this project) 384-dimensional space.
Tokens that behave similarly during training tend to drift toward nearby points in
that space, purely as a side effect of the learning process — nobody hand-designs
these coordinates.

You can't easily picture 384 dimensions, but the 2D/3D intuition transfers: imagine
plotting words on a map where "cat" and "dog" end up near each other (both
animals) and far from "bicycle." Embeddings are that idea, learned automatically,
in a space with far more than three axes to capture far more than one kind of
similarity at once.

> **In this project:** `n_embd = 384` — each of the 251 characters gets a 384-number
> vector, and those numbers themselves are trainable parameters, adjusted throughout
> training just like every other weight in the model.

---

## Module 4: Attention — How the Model Decides What to "Look At"

Here's the core idea that makes the "Transformer" in GPT possible, and the single
most important concept in this primer.

Consider the sentence: *"The robot picked up its tool because it was heavy."*
What does "it" refer to — the robot, or the tool? A human resolves this instantly
using context from earlier in the sentence. A model needs some mechanism to do the
same thing: to let each token look back at *other* tokens and decide which ones are
relevant to interpreting it.

That mechanism is called **self-attention**. For every token, the model computes a
relevance score against every other token in the current context, then blends
information from the most-relevant ones into that token's representation. Crucially,
this happens *for every token simultaneously*, and the relevance scores themselves
are learned during training, not hand-coded.

**Multi-head attention** runs several of these attention computations in parallel,
each initialized differently, so the model can track several kinds of relationships
at once (e.g. one "head" might specialize in pronoun resolution, another in
subject-verb agreement) — like reading a page with several different colored
highlighters, each tracking a different kind of relationship, simultaneously.

This mechanism was introduced in the landmark 2017 paper "Attention Is All You
Need" (Vaswani et al.), which is why this whole family of models is called a
*Transformer*.

> **In this project:** `n_head = 6` — six parallel attention computations per layer.
> `block_size = 256` sets the *context window*: the model can only attend across the
> 256 most recent characters at a time — everything before that is invisible to it
> when predicting the next character.

---

## Module 5: Stacking Layers — Depth and Capacity

One round of attention plus a small amount of additional processing is called a
**layer** (or **block**). A Transformer stacks several of these layers on top of
each other — the output of one layer becomes the input to the next.

Why stack them? Each layer can build on the representations the previous layer
produced. Early layers tend to pick up on simple local patterns (e.g. "a `q` is
almost always followed by a `u`"); later layers can combine those into
higher-level structure (word boundaries, grammatical roles, longer-range context).
More layers generally means more capacity to represent complex patterns — at the
cost of more parameters to train and more compute per prediction.

> **In this project:** `n_layer = 6` — six stacked attention blocks. This is small
> by modern standards (some production models have 100+ layers) — a deliberate
> laptop-scale tradeoff, not a technical ceiling.

---

## Module 6: The Training Loop — How the Model Actually Learns

This is where "training" stops being a metaphor and becomes a concrete, repeatable
loop.

1. **Sample a batch.** Pull a handful of chunks of text from the training data —
   each chunk is `block_size` characters long. (In this project: 64 chunks at a
   time — the `batch_size`.)
2. **Predict.** For every position in every chunk, the model outputs its current
   guess at a probability distribution over "what character comes next."
3. **Measure the loss.** Compare the model's predicted probabilities against the
   *actual* next character (which we know, because it's sitting right there in the
   training text) using a **loss function** — a single number that's large when the
   model's guesses were confidently wrong, and small when they were right. This
   project uses the standard choice for this kind of task, **cross-entropy loss**,
   measured in *nats* (a natural-log-based unit). A useful sanity-check baseline:
   a model that's just guessing uniformly at random among 251 possible characters
   would score a loss of `ln(251) ≈ 5.5` nats — so watching the loss drop below
   that is the first sign the model is actually learning something.
4. **Backpropagate.** Compute, for every one of the model's ~10 million parameters,
   which direction (nudge up or down) and by how much that specific parameter should
   change to have reduced this batch's loss. This calculus — the chain rule applied
   automatically across the entire network — is called **backpropagation**.
5. **Update the weights.** Actually apply those nudges, scaled by a **learning
   rate** (how big a step to take — too big and training becomes unstable and
   diverges; too small and it takes forever to improve). This project also uses a
   **warmup** period (start with tiny steps and ramp up, since the randomly
   initialized model is unreliable at first) followed by **decay** (shrink the step
   size again later in training, for fine adjustments once the model is already
   decent).
6. **Repeat.** One pass through steps 1–5 is one **iteration** (or "step"). This
   project runs 3,000 of them.

Two more concepts that matter for judging whether training actually worked:

- **Train/validation split.** Before training starts, a slice of the corpus (10%,
  in this project) is set aside and *never* used to update the model — only to
  periodically check the loss on text the model hasn't memorized. This guards
  against **overfitting**: a model that's learned to parrot back its exact training
  text instead of learning generalizable patterns would show low loss on training
  data but high loss on this held-out validation data.
- **Checkpoints.** Periodically (and at the end), the model's current parameters
  are saved to disk as a **checkpoint** — a snapshot you can later load back up to
  generate text or resume training, without redoing any of the work that produced it.

> **In this project:** batch size 64, block size 256, 3,000 iterations, loss
> evaluated on the held-out validation set every 250 iterations. See
> [`build-log.md`](build-log.md)'s Task 6 entry for the actual first-vs-last loss
> numbers from this run once training completes.

---

## Module 7: Hardware — Why This Runs on a Laptop at All

Training a language model means doing the same category of matrix arithmetic —
over and over, millions of times — across millions of parameters. CPUs (the general-
purpose chip in every computer) can do this, but slowly, because they're built to
do many different kinds of tasks one after another. **GPUs** (Graphics Processing
Units) were originally built to draw millions of pixels in parallel for video
games, which turns out to be *the same kind of massively-parallel arithmetic* that
neural network training needs — so GPUs became the default hardware for this work.

Apple Silicon Macs (like the M3 used here) have their own built-in GPU, accessed
through Apple's **Metal Performance Shaders (MPS)** framework — PyTorch's `mps`
backend lets training code run on that GPU instead of the CPU, without needing an
NVIDIA GPU or a cloud server.

This matters for setting expectations: production-scale models are trained across
thousands of high-end GPUs running for weeks, at a cost of millions of dollars.
This project's ~10-million-parameter model, training for a few thousand iterations
on a single laptop GPU, is scaled down by roughly six orders of magnitude — which
is exactly the point. It's meant to make the *mechanics* of training visible and
tractable, not to produce a model that competes with production systems.

> **In this project:** `device = 'mps'` in the training config, confirmed available
> (`torch.backends.mps.is_available() == True`) before any training was attempted —
> see [`build-log.md`](build-log.md)'s Task 1 entry.

---

## Module 8: From a Trained Model to Actual Text (Sampling)

Once training produces a checkpoint, generating new text is a different process
from training, called **inference** or **sampling**:

1. Start with a **seed** (a starting fragment of text, possibly empty).
2. Feed it through the model to get a probability distribution over "what
   character comes next."
3. **Sample** one character from that distribution (not necessarily the single
   most likely one — see below).
4. Append it to the text, and feed the *new*, slightly longer text back through the
   model to predict the character after that.
5. Repeat, one character at a time, until you've generated as much text as you
   asked for.

This is why generation is called **autoregressive**: each new character depends on
everything generated so far, feeding back into the model as new input.

A **temperature** parameter controls how "cautious" the sampling is: low
temperature mostly picks the single most-probable character every time (safer, but
repetitive); higher temperature gives lower-probability characters more of a
chance to be picked (more varied, but riskier — more chances to produce nonsense).

> **In this project:** samples are generated from the Task 6 checkpoint via
> nanoGPT's `sample.py`; see `notes/sample_output.txt` (Task 7) for actual
> generated excerpts once that task runs.

---

## Module 9: Putting It All Together

Reading top to bottom, here's the whole pipeline in one pass, module by module:

1. Downloaded 97 public-domain sci-fi books from Project Gutenberg (raw text).
2. **(Module 2)** Tokenized the corpus character-by-character → 251-symbol
   vocabulary, 26.0M training tokens / 2.9M validation tokens.
3. **(Modules 3–5)** Defined a small Transformer: 384-dimensional embeddings,
   6 attention heads, 6 stacked layers, a 256-character context window.
4. **(Module 6)** Trained it for 3,000 iterations, in batches of 64 chunks,
   watching the loss drop from a random-guessing baseline (~5.5 nats) toward
   something meaningfully lower, checked periodically against held-out
   validation data.
5. **(Module 7)** Ran the whole thing on a single Apple M3 laptop GPU via
   PyTorch's `mps` backend — no cloud compute, no cluster.
6. **(Module 8)** Used the trained checkpoint to generate new sci-fi-flavored
   text, one character at a time.

For the specific numbers this run actually produced, see
[`build-log.md`](build-log.md) (the process record) and `pipeline-notes.md`
*(forthcoming)* — the full technical report with cited results and discussion.

---

## Glossary

| Term | Plain-language definition |
|---|---|
| **Token** | A single unit of text the model operates on (here, one character). |
| **Vocabulary** | The complete set of distinct tokens the model knows about. |
| **Embedding** | A learned list of numbers representing a token's "meaning" as a point in a high-dimensional space. |
| **Parameter / weight** | A single learnable number inside the model; training adjusts millions of these. |
| **Attention** | The mechanism letting each token weigh how relevant every other token is to it. |
| **Layer / block** | One round of attention plus additional processing; Transformers stack several. |
| **Loss** | A single number measuring how wrong the model's predictions were; training tries to shrink it. |
| **Backpropagation** | The algorithm that computes how to adjust every parameter to reduce the loss. |
| **Learning rate** | How large a step each parameter update takes. |
| **Iteration / step** | One full pass of predict → measure loss → update weights, on one batch. |
| **Batch** | A handful of training examples processed together in one iteration. |
| **Epoch** | One full pass over the entire training dataset (not directly tracked in this project — progress is measured in iterations instead). |
| **Overfitting** | When a model memorizes training data instead of learning generalizable patterns; caught by comparing training vs. validation loss. |
| **Checkpoint** | A saved snapshot of a model's parameters at a point in training. |
| **Inference / sampling** | Using a trained model to generate new output, as opposed to training it. |
| **Temperature** | A setting controlling how random vs. predictable generated text is. |
| **GPU / MPS** | Parallel-processing hardware well-suited to the matrix math training requires; MPS is Apple Silicon's version. |
| **Fine-tuning** | Continuing to train an already-trained model on a narrower dataset (not done in this project — planned for Phase 2 of the broader learning plan). |
| **BPE (Byte-Pair Encoding)** | The subword tokenization scheme production models like GPT-4 use, instead of this project's character-level approach. |

---

## Further Reading

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
  (2017) — the paper that introduced the Transformer architecture this project uses.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the training engine
  vendored into this repo, and Andrej Karpathy's "Neural Networks: Zero to Hero"
  video series, which nanoGPT was built to accompany.
- [Project Gutenberg](https://www.gutenberg.org/) — the public-domain source of
  every book in this project's training corpus.
