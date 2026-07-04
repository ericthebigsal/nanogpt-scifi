# Training a Character-Level GPT on Public-Domain Science Fiction: Pipeline, Methodology, and Results

*Technical report for `nanogpt-scifi`. Written for readers with a working knowledge of
neural networks and transformers who want a rigorous account of what was built, how,
and what the results do and don't show. For a no-background-required explanation of
the same concepts, see [`llm-primer.md`](llm-primer.md); for a chronological account
of the engineering process, including the incidents summarized in Methodology below,
see [`build-log.md`](build-log.md).*

## Abstract

This project implements and trains a 10.72M-parameter, 6-layer, character-level
GPT from scratch on a 28.9-million-character corpus of 97 public-domain science
fiction works drawn from Project Gutenberg, using Karpathy's nanoGPT as the
training engine and Apple Silicon's MPS backend as the only compute resource. Over
3,000 training iterations, cross-entropy loss fell from 5.4625 to 1.2332 (training
set) and 5.4589 to 1.3032 (held-out validation set), with the two tracking closely
throughout — evidence of learning without significant overfitting at this scale.
Generated samples exhibit consistently correct English orthography, punctuation,
and dialogue conventions, along with vocabulary drawn visibly from the training
corpus, but lack sentence-level semantic coherence — an expected and diagnostic
outcome for a model of this size and tokenization granularity, not a failure of
the pipeline. The project's purpose is pedagogical: to make every stage between
raw text and a trained language model — tokenization, embedding, attention,
optimization — directly inspectable and modifiable, as a deliberate complement to
the increasingly common experience of interacting with language models solely
through an API.

## Motivation

Prompting a hosted large language model teaches a user how to *elicit* behavior
from a model that already exists; it teaches nothing about how that model came to
exist, what its architecture is actually doing at inference time, or what
"training" concretely consists of. This project inverts that relationship. Every
component in the pipeline — the corpus, the tokenizer, the model definition, the
training loop, the sampler — is either authored or vendored and inspected line by
line, at a scale small enough (10.72M parameters, single-laptop-GPU training time
measured in tens of minutes) to iterate on directly rather than treat as a fixed
external service. This is explicitly framed as "going one layer deeper than the
API": a demonstration of engineering fluency with the mechanics underlying modern
language models, rather than fluency with prompting them.

## Background

This section defines, precisely, the components this project's implementation
actually exercises.

**Tokenization.** Neural networks operate on fixed-dimensional numeric tensors, not
raw text; tokenization is the surjective mapping from a string to a sequence of
integers drawn from a fixed vocabulary. This project uses a character-level
vocabulary (Section: Methodology) rather than the byte-pair encoding (BPE) subword
scheme used by GPT-2/GPT-3 and implemented in OpenAI's `tiktoken` (Sennrich et al.,
2016, adapted for LLMs by Radford et al., 2019).

**Embedding.** Each vocabulary entry is associated with a learned, dense vector in
`ℝ^n_embd` (a *token embedding*); a second, separately learned embedding table maps
each sequence position to a vector in the same space (a *positional embedding*, since
the attention mechanism below is otherwise permutation-invariant and has no innate
notion of order). nanoGPT's `GPT` module (`model.py`) sums these two embeddings
elementwise before the first transformer block (`tok_emb + pos_emb`, `model.py:177–178`).

**Self-attention and the Transformer.** The Transformer architecture (Vaswani et
al., 2017) replaces recurrence with *scaled dot-product self-attention*: for each
position in a sequence, a learned linear projection produces a query, key, and
value vector; attention weights are computed as the softmax of scaled query-key dot
products, and the output at each position is a weighted sum of value vectors across
all positions the model is permitted to attend to. *Multi-head* attention runs
several such computations in parallel with independently learned projections, then
concatenates and re-projects the results, allowing different heads to specialize.
Causal (autoregressive) masking restricts each position to attend only to itself
and earlier positions, which is what permits the same forward pass to be used for
both training (next-token prediction at every position simultaneously) and
generation (one new position at a time).

**The training loop, backpropagation, and optimization.** Training minimizes a
scalar loss — here, cross-entropy between the model's predicted next-token
distribution and the true next token, summed/averaged over all positions in a
batch — via stochastic gradient descent. Backpropagation (Rumelhart, Hinton, and
Williams, 1986) computes the gradient of the loss with respect to every parameter
via reverse-mode automatic differentiation; the AdamW optimizer (Loshchilov and
Hutter, 2019) — decoupled-weight-decay Adam — uses those gradients, plus running
first- and second-moment estimates, to update parameters at each step, modulated by
a learning-rate schedule (linear warmup followed by cosine decay in this project's
configuration).

## Methodology

**Corpus construction.** The training corpus is the concatenated body text of 97
public-domain science fiction works, individually downloaded from Project
Gutenberg by eBook ID (`nanoGPT/data/scifi_char/fetch_corpus.py`), with each book's
own Gutenberg boilerplate header/footer stripped via bounded regex matching
against Gutenberg's `*** START/END OF THE PROJECT GUTENBERG EBOOK ***` markers
(`gutenberg_utils.strip_gutenberg_boilerplate`). The list of 97 eBook IDs was
itself extracted from a pre-existing local corpus file via a second, independently
bounded regex over the `[eBook #N]` pattern — a bounded pattern was required
because that source file was discovered to be corrupted (see Limitations and
`build-log.md`'s Task 3 entry for the full incident). Every book's text used for
training was, notwithstanding that discovery, freshly and independently downloaded
from Project Gutenberg, not sourced from the corrupted file. The resulting corpus
totals 28,934,015 characters.

**Tokenization.** A character-level vocabulary was chosen over subword (BPE)
tokenization for two reasons specific to this project's goals: first, it keeps the
encode/decode mapping itself trivially auditable — a sorted-set vocabulary and two
dictionary lookups (`prepare.py`'s `build_vocab`/`encode`/`decode`), with no learned
merge-rule table to separately train and reason about; second, it maximizes the
proportion of total model behavior attributable to the transformer and training
loop rather than to tokenizer design, which is this project's pedagogical focus.
The cost is a longer effective sequence length per unit of text and a token-level
prediction task that is easier per-token (fewer classes) but requires the model to
reconstruct word- and sentence-level structure with no help from the tokenizer.
Scanning the corpus yielded a 251-character vocabulary — larger than the ~65
characters plain English prose would require, due to non-English titles/passages
and Unicode punctuation present in the Gutenberg source texts (accented Latin
characters, Greek letters, and runic characters were all observed in the resulting
`stoi`/`itos` tables). The corpus was split 90/10 into 26,040,613 training and
2,893,402 validation tokens (`np.uint16`-encoded, matching the 251-entry vocabulary's
range) and serialized as `train.bin`/`val.bin`, following nanoGPT's own
`shakespeare_char` convention, plus a `meta.pkl` carrying `vocab_size` and both
lookup tables for use by both `train.py` and `sample.py`.

**Model configuration.** The model (`nanoGPT/config/train_scifi_char.py`) is a
6-layer, 6-head, 384-dimensional Transformer (`n_layer=6, n_head=6, n_embd=384`,
64-dimensional attention heads), with a 256-character context window
(`block_size=256`) and dropout of 0.2, totaling 10.72M parameters (10,811,520
weight-decayed + 4,992 non-decayed, per `train.py`'s parameter accounting). These
values were chosen for laptop-scale feasibility rather than benchmarked
performance: they are roughly two orders of magnitude smaller (by parameter count)
than nanoGPT's own default GPT-2-scale configuration (`n_layer=12, n_head=12,
n_embd=768`, 124M parameters), which targets multi-GPU training this project's
hardware does not have. Weight tying between the token embedding matrix and the
output projection (`lm_head`) — a standard technique per Press and Wolf (2017),
already present in vendored nanoGPT — was left enabled.

**Hardware and the `mps` backend.** Training ran exclusively on a single Apple M3
laptop GPU via PyTorch's Metal Performance Shaders (`mps`) backend
(`device='mps'`), with `compile=False` (PyTorch 2.0 `torch.compile` support for
`mps` was immature enough at implementation time to leave disabled rather than
debug as a secondary variable). One architectural consequence of nanoGPT's own
code is worth noting precisely: `train.py`'s `device_type = 'cuda' if 'cuda' in
device else 'cpu'` classifies `mps` as `'cpu'` for the sole purpose of selecting an
`autocast` context, which disables mixed-precision autocasting (`ctx =
nullcontext()`) and leaves `torch.cuda.amp.GradScaler` self-disabling (it checks
`torch.cuda.is_available()` at construction and emits a `UserWarning` rather than
raising). Net effect: training ran in full float32 precision throughout, which is
the numerically conservative outcome and was not further tuned, since the training
budget here did not make FP16/BF16 throughput a binding constraint.

**Memory constraints and gradient accumulation.** The originally specified
configuration (`batch_size=64, gradient_accumulation_steps=1`, 16,384
tokens/iteration) was found, empirically, to be untrainable in practice on this
project's specific 8GB-RAM development machine: `vm_stat` sampling during a stalled
run showed near-zero free memory and heavy swap activity, and per-iteration time
grew unboundedly (8.2s → 144.3s across the first five iterations) rather than
reaching the sub-second steady state a model this size warrants on `mps`. This was
diagnosed as memory pressure from competing resident processes under Apple's
unified-memory architecture (in which the GPU has no memory pool separate from
system RAM), not a defect in the model or training code — confirmed by a smoke
test showing correct loss values (5.49, matching the theoretical `ln(251)≈5.5`
random-initialization baseline) alongside the pathological timing. The remedy
applied was `batch_size=16` with `gradient_accumulation_steps=4`: four
sequential microbatch forward/backward passes with gradients summed before a
single optimizer step, which reproduces the identical 16,384 tokens/iteration and
therefore the identical training dynamics as the original single-batch
specification, while reducing peak activation memory roughly fourfold (activation
memory scales with microbatch size; parameter and optimizer-state memory are
unaffected by this change). This was preferred over reducing `n_embd`, `n_layer`,
or `block_size`, each of which would have changed the model's effective capacity
or context window rather than merely its memory schedule. Full incident details,
alternatives considered, and the decision rationale are recorded in
`build-log.md`'s Task 6 entry.

**Training-run continuity.** A separate, unrelated interruption occurred during
the corrected run: the training process was terminated by ordinary macOS idle
sleep after reaching iteration 1740 (confirmed via the unified system log, which
showed no memory- or jetsam-related kill near the event, alongside multiple
sleep/wake cycles that session). Because `always_save_checkpoint=True` had
persisted a full resumable checkpoint (model weights, optimizer state, and
`iter_num`) at each `eval_interval` (every 250 iterations), training was resumed
without data loss via nanoGPT's `init_from='resume'` path, wrapped in `caffeinate
-i` to prevent recurrence, and completed the remaining iterations to `max_iters=3000`.

## Results

| Metric | Iteration 0 | Iteration 3000 |
|---|---|---|
| Training loss (cross-entropy, nats) | 5.4625 | **1.2332** |
| Validation loss (cross-entropy, nats) | 5.4589 | **1.3032** |

Full loss trajectory at each `eval_interval` (250 iterations):

```
step    0: train 5.4625, val 5.4589
step  250: train 2.1241, val 2.1001
step  500: train 1.7561, val 1.7495
step  750: train 1.5712, val 1.5959
step 1000: train 1.4663, val 1.4940
step 1250: train 1.4077, val 1.4349
step 1500: train 1.3500, val 1.3964
step 1750: train 1.3125, val 1.3600
step 2000: train 1.2921, val 1.3437
step 2250: train 1.2602, val 1.3339
step 2500: train 1.2465, val 1.3182
step 2750: train 1.2475, val 1.3128
step 3000: train 1.2332, val 1.3032
```

The loss reduction is steepest in the first 500 iterations (5.46 → 1.76, roughly
68% of the total nats-of-loss reduction achieved over the full run) and continues
to improve, with diminishing returns, through iteration 3000. The final
training/validation gap (0.071 nats) is small relative to the total loss reduction
achieved (4.16 nats), indicating the model is not meaningfully overfitting the
training split at this parameter count, iteration budget, and dropout rate (0.2).

Dataset scale for reference: 28,934,015 total characters, 251-character
vocabulary, 26,040,613 training tokens, 2,893,402 validation tokens (Methodology).
Total wall-clock time across both training attempts (including the memory-pressure
diagnosis and the sleep-interruption/resume described in Methodology) was
approximately 104 minutes; steady-state per-iteration compute time, excluding
those two incidents, was consistently 0.9–1.1 seconds/iteration throughout, for an
estimated ~50 minutes of pure computation had the run proceeded uninterrupted.

Representative generated sample (temperature 0.8, 500 characters, from
`notes/sample_output.txt`):

> "position, under the beginning of the earth, may be better of the earth of
> the brute explosion of the last merely confident provinces."
>
> "Perhaps," said the control body. "I will go out alone of your own mother!"
>
> "Well?"
>
> "Come for me," he said, "that this activity-men is utterly in about the
> monster?"
>
> "Well," said the man. "Now let us which well be one man we shall say to get
> out off of the Brazing—or they come after an idle added to disappear."
>
> "What do you can be the prince, Pencroft?"

Orthography, punctuation, capitalization, and quoted-dialogue formatting are
consistently well-formed throughout all three generated samples. Vocabulary
plausibly drawn from the training corpus's genre is visible (*monster*,
*explosion*, *Pencroft* — a recognizable character name from Jules Verne's *The
Mysterious Island*, one of the 97 source texts). Sentence-level semantic
coherence is absent (e.g. "this activity-men is utterly in about the monster").

## Discussion

The results are consistent with a well-understood property of small,
character-level language models: cross-entropy loss and local (sub-sentence)
fluency both improve substantially and measurably with training, while
sentence-level and discourse-level coherence require either substantially more
parameters, substantially more training data/compute, or a tokenization scheme
that removes the burden of re-deriving word and clause boundaries at every
generation step (subword/BPE tokenization, as used by GPT-2 and later models,
shifts exactly this burden into the tokenizer). This project's 10.72M-parameter,
character-level configuration was chosen specifically to make that character-to-word-to-sentence
gap visible and attributable, rather than to maximize output quality — a
different design goal from a subword-tokenized, order-of-magnitude-larger model
that would likely produce locally *and* globally more coherent text, at the cost
of both training time this project's hardware cannot support and reduced
visibility into the tokenizer's own contribution to that coherence.

The engineering incidents described in Methodology (memory pressure,
sleep-interruption) are treated here as part of the technical record rather than
appendix material, because both directly shaped implementation decisions
(`gradient_accumulation_steps`, `caffeinate`-wrapped, checkpoint-resumable
invocation) that a reader reproducing this pipeline on comparable consumer
hardware would need to replicate deliberately, not incidentally.

Neither fine-tuning an existing pretrained model nor BPE tokenization was in
scope for this phase of work; both are deliberately deferred (see the project's
broader learning plan) to a later phase that specifically investigates
fine-tuning and benchmarking against pretrained baselines, for which this
from-scratch pipeline serves as a mechanistic foundation rather than a
competing approach.

## Limitations

- **Scale.** 10.72M parameters and 3,000 training iterations are small relative
  to any production language model by several orders of magnitude; results
  should not be read as evidence about scaling behavior beyond this regime.
- **Compute budget.** Training ran on a single consumer laptop GPU under real
  memory constraints (Methodology); no hyperparameter search was performed over
  learning rate, architecture width/depth, or context length — the configuration
  reported is the first laptop-feasible configuration evaluated, not a tuned one.
- **Tokenization ceiling.** Character-level tokenization was a deliberate choice
  (Methodology) but is also a hard ceiling on achievable sample coherence at fixed
  parameter count and training budget, as discussed above.
- **Corpus provenance caveat.** The list of source eBook IDs was recovered from a
  corrupted local file via a narrowly bounded extraction pattern (Methodology);
  while every book's actual text was independently re-downloaded and verified
  (Task 4), the original file's corruption is a reminder that this project's
  reproducibility depends on Project Gutenberg's IDs and hosted texts remaining
  stable, not on any local intermediate artifact.
- **Single run.** Results reflect one training run at one fixed random seed;
  no variance estimate across seeds or reruns is reported.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
  Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* Advances in
  Neural Information Processing Systems 30. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).
  *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI.
- Sennrich, R., Haddow, B., & Birch, A. (2016). *Neural Machine Translation of
  Rare Words with Subword Units.* Proceedings of ACL 2016. (Byte-pair encoding,
  the tokenization scheme this project deliberately does not use — Methodology.)
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning
  Representations by Back-Propagating Errors.* Nature, 323(6088), 533–536.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*
  (AdamW). International Conference on Learning Representations.
  [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Press, O., & Wolf, L. (2017). *Using the Output Embedding to Improve Language
  Models.* Proceedings of EACL 2017. (Weight tying, used unmodified from vendored
  nanoGPT.)
- Karpathy, A. [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT license) — the
  training engine vendored into this repo, and the accompanying "Neural Networks:
  Zero to Hero" video series.
- Project Gutenberg. [https://www.gutenberg.org/](https://www.gutenberg.org/) —
  source of all 97 public-domain works comprising the training corpus.
