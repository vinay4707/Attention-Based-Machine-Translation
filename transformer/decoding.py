"""Beam search decoding for the encoder-decoder Transformer.

Implements standard beam search with length normalization. Works with
any model that exposes encode() and decode() methods matching the
Transformer interface.

Usage:
    from transformer.decoding import beam_search

    sequences, scores = beam_search(
        model, src, bos_idx=2, eos_idx=3,
        beam_size=4, max_len=100, length_penalty=0.6,
    )
    best = sequences[0]  # highest-scoring decoded token ids
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def beam_search(
    model,
    src: Tensor,
    bos_idx: int,
    eos_idx: int,
    beam_size: int = 4,
    max_len: int = 100,
    length_penalty: float = 0.6,
) -> tuple[list[list[int]], list[float]]:
    """Beam search decoding for a single source sequence.

    Runs beam search for one example (batch size 1). Returns the
    `beam_size` completed hypotheses sorted by length-normalized score
    (highest first). Hypotheses that never emit EOS are included at
    max_len with their accumulated score.

    Length normalization follows Wu et al., 2016:
        score = log_prob_sum / ((5 + len) / 6) ** alpha

    Args:
        model: Transformer with encode(src, src_mask) and
            decode(tgt, memory, tgt_mask, src_mask) methods plus
            make_src_mask, make_tgt_mask, and output_proj.
        src: Source token ids, shape (1, src_len).
        bos_idx: Beginning-of-sequence token id.
        eos_idx: End-of-sequence token id.
        beam_size: Number of active hypotheses to maintain.
        max_len: Maximum number of generated tokens (not counting BOS).
        length_penalty: Alpha for length normalization (0 = no penalty).

    Returns:
        Tuple of (sequences, scores) where:
            sequences: List of token id lists (EOS included if generated,
                BOS excluded), sorted best-first.
            scores: Length-normalized log-probability for each sequence.
    """
    model.eval()
    device = src.device

    with torch.no_grad():
        src_mask = model.make_src_mask(src)                  # (1, 1, 1, S)
        memory = model.encode(src, src_mask)                  # (1, S, d_model)

        # Expand to beam_size copies
        memory = memory.expand(beam_size, -1, -1)             # (B, S, d_model)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)     # (B, 1, 1, S)

        # Active beams: list of (token_ids, cumulative_log_prob)
        beams: list[tuple[list[int], float]] = [([bos_idx], 0.0)]
        completed: list[tuple[list[int], float]] = []

        for _ in range(max_len):
            if len(beams) == 0:
                break

            # Pad all current beams to the same length for batch decode
            n_active = len(beams)
            max_t = max(len(b[0]) for b in beams)
            tgt_ids = torch.zeros(n_active, max_t, dtype=torch.long, device=device)
            for i, (ids, _) in enumerate(beams):
                tgt_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            # Decode one step
            mem_slice = memory[:n_active]      # (n_active, S, d)
            mask_slice = src_mask[:n_active]   # (n_active, 1, 1, S)
            tgt_mask = model.make_tgt_mask(tgt_ids)
            dec_out = model.decode(tgt_ids, mem_slice, tgt_mask, mask_slice)
            logits = model.output_proj(dec_out[:, -1, :])  # (n_active, vocab)
            log_probs = F.log_softmax(logits, dim=-1)       # (n_active, vocab)

            # Expand candidates
            vocab_size = log_probs.size(-1)
            candidates: list[tuple[list[int], float]] = []
            for i, (ids, cum_lp) in enumerate(beams):
                # Take top beam_size tokens per beam
                topk_lp, topk_ids = log_probs[i].topk(beam_size)
                for lp, tok in zip(topk_lp.tolist(), topk_ids.tolist()):
                    candidates.append((ids + [tok], cum_lp + lp))

            # Sort by current score and keep top beam_size
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[: beam_size * 2]

            next_beams: list[tuple[list[int], float]] = []
            for ids, cum_lp in candidates:
                if ids[-1] == eos_idx:
                    completed.append((ids, cum_lp))
                else:
                    next_beams.append((ids, cum_lp))
                if len(next_beams) >= beam_size:
                    break

            beams = next_beams

            # Early stop if we have enough completed hypotheses
            if len(completed) >= beam_size:
                break

        # Move any surviving beams into completed
        completed.extend(beams)

        # Length-normalize and sort
        def _score(item: tuple[list[int], float]) -> float:
            ids, cum_lp = item
            length = len(ids) - 1  # exclude BOS
            norm = ((5 + max(length, 1)) / 6) ** length_penalty
            return cum_lp / norm

        completed.sort(key=_score, reverse=True)

        sequences = [ids[1:] for ids, _ in completed]  # strip BOS
        scores = [_score(item) for item in completed]

        return sequences, scores
