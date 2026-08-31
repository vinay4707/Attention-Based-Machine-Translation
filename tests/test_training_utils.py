"""Tests for label_smoothed_ce and the Noam learning rate schedule."""

import math

import pytest
import torch
import torch.nn as nn

from transformer.scheduler import noam_schedule, get_noam_scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_smoothed_ce(logits, targets, smoothing=0.1, pad_idx=1):
    """Inline copy of train.label_smoothed_ce for isolated testing."""
    vocab_size = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, smoothing / (vocab_size - 2))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
        smooth_targets[:, pad_idx] = 0.0
        mask = targets == pad_idx
        smooth_targets[mask] = 0.0

    loss = -(smooth_targets * log_probs).sum(dim=-1)
    n_tokens = (~mask).sum()
    return loss.sum() / n_tokens.clamp(min=1)


# ---------------------------------------------------------------------------
# Label smoothing tests
# ---------------------------------------------------------------------------

class TestLabelSmoothing:
    def test_returns_scalar(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(8, 20)
        targets = torch.randint(2, 20, (8,))
        loss = _label_smoothed_ce(logits, targets, smoothing=0.1)
        assert loss.shape == ()
        assert loss.dim() == 0

    def test_non_negative(self) -> None:
        torch.manual_seed(1)
        logits = torch.randn(16, 50)
        targets = torch.randint(2, 50, (16,))
        loss = _label_smoothed_ce(logits, targets, smoothing=0.1)
        assert loss.item() >= 0.0

    def test_zero_smoothing_matches_cross_entropy(self) -> None:
        """smoothing=0 should match nn.CrossEntropyLoss on non-pad tokens."""
        torch.manual_seed(2)
        vocab_size = 30
        pad_idx = 1
        N = 12
        logits = torch.randn(N, vocab_size)
        targets = torch.randint(2, vocab_size, (N,))  # no pad tokens

        ce = nn.CrossEntropyLoss(ignore_index=pad_idx)
        expected = ce(logits, targets).item()
        actual = _label_smoothed_ce(logits, targets, smoothing=0.0, pad_idx=pad_idx).item()
        assert abs(actual - expected) < 1e-5, f"smoothing=0: {actual:.6f} vs CE: {expected:.6f}"

    def test_pad_tokens_excluded(self) -> None:
        """Loss is the same whether pad tokens are appended or not."""
        torch.manual_seed(3)
        vocab_size = 20
        pad_idx = 1
        logits_base = torch.randn(4, vocab_size)
        targets_base = torch.randint(2, vocab_size, (4,))

        # Add pad rows
        pad_logits = torch.randn(2, vocab_size)
        pad_targets = torch.full((2,), pad_idx)
        logits_padded = torch.cat([logits_base, pad_logits], dim=0)
        targets_padded = torch.cat([targets_base, pad_targets], dim=0)

        loss_base = _label_smoothed_ce(logits_base, targets_base, pad_idx=pad_idx)
        loss_padded = _label_smoothed_ce(logits_padded, targets_padded, pad_idx=pad_idx)
        assert torch.allclose(loss_base, loss_padded, atol=1e-5), (
            f"pad rows changed loss: {loss_base:.6f} vs {loss_padded:.6f}"
        )

    def test_smoothing_increases_entropy(self) -> None:
        """Higher label smoothing should increase loss on a near-perfect prediction."""
        torch.manual_seed(4)
        vocab_size = 50
        # Construct logits with a clear winner
        logits = torch.full((8, vocab_size), -10.0)
        targets = torch.ones(8, dtype=torch.long) * 5
        logits[:, 5] = 10.0  # very confident

        loss_no_smooth = _label_smoothed_ce(logits, targets, smoothing=0.0, pad_idx=1)
        loss_smooth = _label_smoothed_ce(logits, targets, smoothing=0.2, pad_idx=1)
        assert loss_smooth.item() > loss_no_smooth.item(), (
            "label smoothing should raise loss on overconfident predictions"
        )

    def test_all_pad_returns_finite(self) -> None:
        """All-pad batch: loss should not be nan or inf (clamp prevents div-by-zero)."""
        torch.manual_seed(5)
        pad_idx = 1
        logits = torch.randn(4, 20)
        targets = torch.full((4,), pad_idx)
        loss = _label_smoothed_ce(logits, targets, smoothing=0.1, pad_idx=pad_idx)
        assert torch.isfinite(loss), f"all-pad loss should be finite, got {loss.item()}"

    def test_single_token(self) -> None:
        """Single non-pad token: loss should be a positive scalar."""
        torch.manual_seed(6)
        logits = torch.randn(1, 10)
        targets = torch.tensor([3])
        loss = _label_smoothed_ce(logits, targets, smoothing=0.1, pad_idx=1)
        assert loss.item() > 0.0
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Noam schedule tests
# ---------------------------------------------------------------------------

class TestNoamSchedule:
    def test_peak_at_warmup_steps(self) -> None:
        """lr peaks near step == warmup_steps."""
        d_model = 512
        warmup = 4000
        # The Noam schedule transitions from increasing to decreasing at step=warmup_steps
        # warmup^{-0.5} == warmup * warmup^{-1.5} at that exact point
        at_peak = noam_schedule(warmup, d_model, warmup)
        just_before = noam_schedule(warmup - 1, d_model, warmup)
        just_after = noam_schedule(warmup + 1, d_model, warmup)
        # at_peak >= just_before and at_peak >= just_after
        assert at_peak >= just_before - 1e-9
        assert at_peak >= just_after - 1e-9

    def test_lr_increases_during_warmup(self) -> None:
        """Schedule is monotonically increasing for step < warmup_steps."""
        d_model = 256
        warmup = 1000
        scales = [noam_schedule(s, d_model, warmup) for s in range(1, warmup)]
        diffs = [scales[i + 1] - scales[i] for i in range(len(scales) - 1)]
        assert all(d >= 0 for d in diffs), "schedule decreased before warmup_steps"

    def test_lr_decreases_after_warmup(self) -> None:
        """Schedule is monotonically decreasing for step > warmup_steps."""
        d_model = 256
        warmup = 500
        steps = range(warmup + 1, warmup + 1000)
        scales = [noam_schedule(s, d_model, warmup) for s in steps]
        diffs = [scales[i + 1] - scales[i] for i in range(len(scales) - 1)]
        assert all(d <= 0 for d in diffs), "schedule increased after warmup_steps"

    def test_step_zero_is_safe(self) -> None:
        """Step=0 is clamped to 1 internally, should not raise."""
        scale = noam_schedule(0, 512, 4000)
        assert math.isfinite(scale)
        assert scale > 0.0

    def test_scale_positive(self) -> None:
        """Scale is always positive."""
        for step in [1, 100, 4000, 10000, 100000]:
            scale = noam_schedule(step, 512, 4000)
            assert scale > 0.0, f"negative scale at step {step}"

    def test_larger_d_model_gives_smaller_scale(self) -> None:
        """d_model^{-0.5} means larger d_model => smaller scale at same step."""
        step, warmup = 500, 4000
        scale_small = noam_schedule(step, 128, warmup)
        scale_large = noam_schedule(step, 512, warmup)
        assert scale_small > scale_large

    def test_formula_matches_paper(self) -> None:
        """Verify the exact formula from Vaswani et al. at a known step."""
        d_model = 512
        warmup = 4000
        step = 100  # step < warmup, so formula = d_model^{-0.5} * step * warmup^{-1.5}
        expected = (d_model ** -0.5) * step * (warmup ** -1.5)
        actual = noam_schedule(step, d_model, warmup)
        assert abs(actual - expected) < 1e-10

    def test_get_noam_scheduler_steps_lr(self) -> None:
        """Scheduler changes lr correctly across a few optimizer steps."""
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
        scheduler = get_noam_scheduler(optimizer, d_model=64, warmup_steps=100)

        lrs = []
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        # During warmup (step 1..5 << 100), lr should be increasing
        assert lrs[-1] > lrs[0], "lr should increase during warmup phase"
        # All lrs should be finite and positive
        assert all(lr > 0 and math.isfinite(lr) for lr in lrs)
