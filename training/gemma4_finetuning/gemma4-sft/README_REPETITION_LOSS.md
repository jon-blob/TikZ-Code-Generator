# Repetition-aware SFT loss

This project keeps the native Unsloth/TRL cross-entropy path and adds a
memory-efficient repetition regularizer by masking repeated target n-gram
positions before CE is computed.

It is intentionally **not** a full-logits unlikelihood loss. The earlier
full-logits custom loss can materialize `[seq_len, vocab_size]` logits and use
much more VRAM. This implementation leaves the model's native fused/chunked CE
path intact.

## Default behavior

Defaults work even if these fields are absent from `training_config.py`:

```python
repetition_loss_enabled = True
repetition_ngram_orders = (16,)
repetition_free_occurrences = 2
repetition_keep_every = 4
repetition_protect_first_tokens = 32
repetition_protect_last_tokens = 64
repetition_max_mask_fraction = 0.35
```

Interpretation:

- First two occurrences of an exact 16-token n-gram are trained normally.
- From the third occurrence onward, positions become repetition candidates.
- One out of four candidates remains supervised; the others are set to `-100`.
- At most 35% of active completion targets may be masked.
- The first 32 and last 64 completion tokens are always protected.
- Evaluation loss uses the original labels, so it stays comparable with the
  baseline run.

## Recommended first A/B test

Baseline:

```python
repetition_loss_enabled = False
```

Regularized:

```python
repetition_loss_enabled = True
repetition_ngram_orders = (16,)
repetition_free_occurrences = 2
repetition_keep_every = 4
repetition_max_mask_fraction = 0.35
```

Keep learning rate, dataset, seed, dropout, weight decay and generation config
identical between both runs.

## Debug output

`<output_dir>/generations/repetition_loss_debug.jsonl`

Each training microbatch logs:

- `active_tokens`
- `candidate_tokens`
- `masked_tokens`
- `masked_fraction`
- per-sample equivalents

If `masked_fraction` is almost always zero, exact 16-gram repetition is not
common enough and you can test `(8, 16)`. If it is frequently near 0.35, the
regularizer is hitting its safety cap and should not be made stronger before
checking free-generation behavior.
