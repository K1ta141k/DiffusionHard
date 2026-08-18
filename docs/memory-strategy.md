# Inference memory and cache strategy

## Decision summary

There is no single "KV cache" decision. Diffusion inference has four distinct
reuse objects with different correctness and lifetime rules:

| Object | Lifetime | Exact reuse rule | Recommended placement |
| --- | --- | --- | --- |
| Quantized weights | Model deployment | Always reusable | External DDR; tile through SRAM |
| Prompt/context K/V | Request | Exact only when prompt states cannot attend to the changing generation region | External DDR; current tile in SRAM |
| Active-block K/V and activations | Layer/model evaluation | Recompute after any input change for the exact full-attention baseline | SRAM scratchpad; overwrite by layer |
| Denoising probabilities/candidates | Until input tokens change | Exact for a non-time-conditioned checkpoint | Compact request-local cache |

Our default serving policy should be:

1. Allocate request-local state when a request begins.
2. Prefill once and retain prompt K/V only for a model with an attention mask
   that makes prefix reuse exact.
3. Recompute active-block K/V for every model evaluation.
4. Skip the entire model and reuse denoising probabilities when no token changed.
5. Release prompt K/V, active state, and probability/candidate state immediately
   when the request completes.

Cross-request prefix caching is a later serving feature. It requires an explicit
capacity policy, prompt identity/hash, tenant isolation, and privacy review; it
must not be the default.

## Multi-turn conversation policy

A conversation session can span multiple generation requests. State should be
split into two tiers:

| Session state | Keep after an answer? | Reason |
| --- | --- | --- |
| Transcript token IDs and turn metadata | Yes | Tiny and always sufficient to recompute |
| Prompt/completed-turn K/V | Opportunistically | Large; reusable only under prefix-safe attention |
| Active response-block K/V/activations | No | Invalid after the answer completes |
| DDPM probability/candidate cache | No | A follow-up changes the input immediately |

For original full-attention LLaDA, appending a follow-up can change the hidden
state of every earlier token. Previously stored K/V is therefore not an exact
cache of the new sequence. The exact serving baseline retains the transcript
tokens and recomputes all layer state on the follow-up.

For a block-causal or prefix-isolated serving model, completed turns form a
frozen prefix. In that case, retain their K/V across follow-ups, prefill only the
new user tokens, denoise the new response block, and append the completed block
to the session cache. This is the preferred production contract, but it must be
supported or validated by the model rather than imposed silently on vanilla
LLaDA.

The MDLM macOS harness now demonstrates this execution contract with the real
169.6M-parameter checkpoint. It prevents prefix queries from attending to the
active suffix, captures per-layer prefix K/V, and then evaluates only the suffix
against that cache. Across five suffix edits, the cached and fully recomputed
prefix-isolated paths had 100% suffix top-1 agreement and maximum normalized
logit RMSE below `9.3e-7`. The original checkpoint was not trained with this
mask, so this proves execution equivalence, not retained language quality.

A subsequent held-out WikiText-2 reconstruction gate provides preliminary
quality evidence. Across 10,752 scored token decisions, prefix isolation reduced
top-1 accuracy by 2.46% relative while improving mean negative log likelihood by
0.61%. It passed the predefined 5% limits for relative accuracy and loss. This
does not replace a complete DDPM generation evaluation, but it is sufficient to
advance the cache policy into a conditioned sampler experiment.

The conditioned sampler now confirms exact end-to-end cache replay on five
64-step continuations. Cached and recomputed prefix-isolated paths produced the
same tokens and transition schedules in all five cases. Median model-forward
time fell by 51.6%, and median sampling wall time fell by 46.0%, after a 22.7 ms
one-time prefix prefill. None of the five completed outputs left valid terminal
K/V, confirming that a separate finalization pass is required before retaining
the generated answer for a follow-up request.

The completed-boundary implementation now closes that gap. In three measured
two-turn sessions with 64 initial tokens, 16 first-answer tokens, 16 follow-up
tokens, and 16 second-answer tokens, the retained-cache path matched full
block-causal recomputation exactly on both answers and every DDPM transition.
For the second answer, median model-forward time fell from 729.5 ms to 279.9 ms.
Including first-answer finalization and follow-up materialization, the cached
cross-request cost was 305.5 ms, still 58.1% below recomputation. Finalizing the
second answer as well, so the cache was ready for a third turn, cost 322.8 ms
and retained a 55.8% reduction. The final 112-token FP32 session cache occupied
7.875 MiB.

This evidence changes the serving recommendation for a prefix-safe model:
retain completed-turn K/V for a likely near-term follow-up, subject to TTL and
LRU capacity control. It does not make conventional K/V reuse exact for the
original full-attention LLaDA architecture.

The bounded lifecycle model adds an important qualification. For four measured
MDLM-shaped two-turn sessions, caching within each request reduced model token
positions from 12,288 to 2,688 relative to full recomputation. Retaining K/V
across requests reduced the ready-for-another-turn total only from 2,688 to
2,496, an additional 7.1%. If the final answers are not eagerly finalized after
their last observed request, the incremental reduction is 9.5%. The large
previous latency result mostly proves the value of prefix isolation during the
denoising loop. Cross-request retention is a smaller second optimization.

Therefore, always perform request-local prefix caching for a prefix-safe model.
Retain that K/V after the request only when terminal K/V is already available,
the session is likely to receive a near-term follow-up, or measured prefill cost
justifies eager finalization. A TTL/LRU manager prevents cold conversations from
turning a modest latency benefit into permanent DDR pressure.

There is an additional diffusion-specific cost: the last denoising evaluation
may have consumed a canvas that still contained masks and then changed tokens.
Its K/V is not necessarily the K/V of the finished answer. Exact retention then
requires a final forward pass over the completed response block. If the sampler
already performs a terminal evaluation on the final tokens, this pass can be
eliminated. The serving trace must record which case occurred.

For a first turn with `P` prompt tokens and `A` answer tokens, eager answer-K/V
materialization costs approximately `A` token positions and avoids recomputing
`P + A` positions if a follow-up arrives. Ignoring DDR pressure, it breaks even
when follow-up probability exceeds `A / (P + A)`. A long answer after a short
prompt is therefore a poor candidate for eager retention unless terminal K/V is
already available.

Use a request-cache manager rather than permanent allocation:

1. Key entries by session, model revision, tokenizer revision, precision, and
   attention policy.
2. Keep hot-session K/V in external DDR with an inactivity TTL and LRU eviction.
3. Retain token IDs after K/V eviction so a follow-up can recompute correctly.
4. Evict K/V under memory pressure before model weights or active-request state.
5. Delete all session state on conversation deletion, logout, or privacy-policy
   expiry; never reuse one user's K/V for another user.

The retention decision should compare expected saved prefill time with cache
size and follow-up probability. Large full-MHA models make this especially
important: a 32-layer, hidden-size-4096 FP16 model uses roughly 512 KiB of K/V
per cached token, or about 512 MiB for a 1,024-token conversation. A 4 GB board
can retain only a few such sessions after weights and runtime buffers, so
unbounded session KV retention is not viable.

## Why autoregressive KV caching does not transfer directly

In a causal autoregressive model, old tokens cannot attend to a new token. Their
layer K/V values therefore remain valid as decoding advances.

MDLM and original LLaDA use bidirectional/full attention over the denoised
sequence. When any token changes, every token can observe that change at the
next layer. Even an unmasked token's hidden representation can therefore drift.
Reusing its K/V across denoising steps is not generally exact.

This gives three cases:

- **MDLM-OWT unconditional generation:** there is no prompt prefill to cache.
  Recompute per model evaluation; use exact probability caching across no-change
  transitions.
- **Original LLaDA full-attention prompting:** use recompute-all as the exact
  baseline. Standard prompt KV reuse is an approximation unless the model or
  attention mask isolates the prompt from the changing response.
- **Block/prefix-isolated diffusion model:** prefill the fixed prefix once, keep
  its K/V for the request, recompute the bidirectional active block, and retain
  completed-turn K/V opportunistically for near-term follow-ups.

Recent dLLM cache methods such as dKV-Cache, FreeCache, MaskKV, and Elastic-Cache
reuse selected tokens/layers approximately and validate the resulting quality.
They are valuable experiment points, not assumptions to bake into the exact
baseline.

## Prefill and decode speed model

For the same dense Transformer and prompt length, AR and diffusion prefill are
similar parallel matrix workloads. LLaDA's full bidirectional attention can do
more attention work than causal prefill, but projection and feed-forward matrix
multiplications still dominate at ordinary lengths. There is no inherent
diffusion prefill speedup.

The major difference appears after prefill. Let `P` be prompt tokens, `B` the
generation block, `T` AR output tokens, `S` diffusion transitions, and `E` real
diffusion model evaluations after exact probability-cache hits are removed.

| Execution | Approximate token-layer work | Full weight sweeps |
| --- | ---: | ---: |
| AR with KV | `P + T` | `1 + T` |
| Vanilla full-attention diffusion | `S × (P + B)` | `S` |
| Diffusion with exact no-change caching | `E × (P + B)` | `E` |
| Prefix-isolated block diffusion | `P + E × B` | model dependent |

AR decode performs little parallel token work per weight sweep, so it tends to
be bandwidth bound. Diffusion performs a whole block per sweep, increasing
matrix utilization and arithmetic intensity, but it may repeat that large
computation many times. Diffusion wins only when fewer model evaluations and
better utilization outweigh the repeated full-block work.

Our 64-token MDLM run illustrates both sides:

- Uncached DDPM used 64 full model evaluations and took 1.253 seconds on MPS.
- Exact probability caching reduced this to 38 evaluations and 0.704 seconds.
- This is still 38 full 64-position passes, not the equivalent of 38 cheap AR
  single-token decodes.

Therefore, baseline LLaDA is generally slower than a comparable cached AR model.
Its speed opportunity comes from few-step decoding, block/prefix isolation,
exact no-change skipping, approximate K/V refresh, and high utilization on
otherwise under-filled hardware, not from a cheaper prefill.

## MDLM-OWT working-set sizes

For the current 12-layer, hidden-size-768 checkpoint and a 64-token canvas:

| Working set | FP16 bytes | FPGA implication |
| --- | ---: | --- |
| One layer K/V | 192 KiB | Fits comfortably on chip |
| K/V for all 12 layers | 2.25 MiB | Technically fits, but consumes most K26 SRAM |
| One hidden-state buffer | 96 KiB | Double buffering is inexpensive |
| One QKV buffer | 288 KiB | Fits as a layer-local scratchpad |
| Full vocabulary probabilities | 6.14 MiB | Does not fit in K26 on-chip SRAM |
| All FP32 model weights | 678.5 MB | Must remain in external DDR |
| All INT8 model weights | 169.6 MB | Must remain in external DDR |

The result is counterintuitive but important: at this model and canvas size,
conventional K/V capacity is not the primary memory problem. Weight streaming
and the 50,258-way vocabulary output dominate.

## Replace a full probability cache with candidate state

For a DDPM-cache transition with unchanged input, the cached distribution has
two parts:

1. A scalar decision to stay masked or become unmasked.
2. Conditional on unmasking, a token sampled from the unchanged model
   distribution.

The token candidate is independent of the transition at which unmasking occurs.
This gives a distribution-equivalent hardware formulation:

1. On a real model evaluation, sample one candidate token per masked position.
2. Store the candidate token ID plus small scheduling metadata.
3. On cache-hit transitions, perform only the scalar mask/unmask decision.
4. Invalidate all remaining candidates as soon as any input token changes and a
   new model evaluation becomes necessary.

The factorization now passes both an analytic and statistical gate. Across 64
transitions and 64 independently generated distributions, maximum float64
probability error was `1.11e-16`. A separate 500,000-draw experiment measured
total-variation distance `0.003102`, below the predefined `0.01` limit.

For 64 MDLM positions, 16-bit token IDs plus active and valid bitmaps require
144 bytes rather than a 6.135 MiB FP16 probability tensor, a 44,674x reduction.
The real checkpoint also completed a 64-transition candidate-cache run with 41
model evaluations and 23 compact-cache hits. Because the random stream is
factorized differently, output is distribution-equivalent rather than
seed-for-seed identical to the original exponential-race implementation.

## Pre-RTL simulator matrix

The simulator should compare the following policies before the standalone
kernel interface is frozen:

| Policy | Correctness class | What it tests |
| --- | --- | --- |
| Recompute all K/V | Exact | Reference latency and traffic |
| Probability cache on no-change transition | Exact | Already measured model-call elimination |
| Candidate-token cache | Distribution-equivalent candidate | Removes full probability storage on cache hits |
| Request-local prompt KV | Exact only for prefix-isolated attention | Prefill-once versus repeated prompt work |
| Refresh K/V every N steps | Approximate | Speed/quality curve |
| Layer-selective KV refresh | Approximate | Shallow versus deep representation drift |
| Dirty-token/token-selective refresh | Approximate | Exploit edit sparsity |
| Compressed/evicted prompt KV | Approximate | Long-context capacity and bandwidth |

For every policy, report:

- prefill latency and bytes,
- denoising latency and bytes,
- peak request memory,
- model evaluations and cache hits,
- generated tokens per second,
- exact token agreement where applicable,
- task accuracy or perplexity for approximate policies.

## FPGA architecture consequence

Do not build a fixed, all-layer KV-cache SRAM before these experiments. The
first FPGA memory system should expose:

- external-DDR DMA for weights and request-local state,
- a banked, dynamically partitioned SRAM scratchpad,
- layer-local K/V and activation buffers,
- active-token bitmap and candidate-token storage,
- explicit invalidate/commit control,
- counters for bytes, cache hits, stalls, and occupancy.

This structure supports the exact baseline, prompt caching when valid, and later
approximate KV policies without committing the RTL to one model's cache rule.
