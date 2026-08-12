"""Pure-torch PPO/GRPO core for the embedding-policy bandit — ported from the text experiment
(trl.experimental.embedding_ppo.modeling). The policy is softmax(sim(q,d)/temperature) over an
N-candidate pool per query; episode length 1, so GAE collapses to A = R − V(q).

Only torch imports here so the math is unit-testable without transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """V(q): linear head on the (detached) query embedding, kept in fp32 for stability."""

    def __init__(self, hidden_size, init_bias=0.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.linear.bias.fill_(float(init_bias))

    def forward(self, q_emb):
        return self.linear(q_emb.detach().float()).squeeze(-1)


def pool_logps(q, d, temperature):
    """q: (B, dim) normalized; d: (B, P, dim) normalized -> log-softmax over the pool (B, P)."""
    sims = torch.einsum("bd,bpd->bp", q, d) / temperature
    return F.log_softmax(sims, dim=-1), sims


def pl_sample_lists(scaled_sims, k, n_lists):
    """Sample ordered k-lists from the Plackett-Luce model via Gumbel-top-k.

    scaled_sims: (B, M) = sim/temperature over the candidate support.
    Returns (B, G, k) indices. Sorting Gumbel-perturbed scores is EXACTLY sequential
    softmax sampling without replacement (Gumbel-Plackett-Luce equivalence), so these
    are unbiased draws from the factorized policy pi(D|q) = prod_i softmax_remaining(s).
    """
    B, M = scaled_sims.shape
    u = torch.rand(B, n_lists, M, device=scaled_sims.device).clamp_(1e-20, 1 - 1e-7)
    gumbel = -torch.log(-torch.log(u))
    return (scaled_sims.unsqueeze(1) + gumbel).topk(k, dim=-1).indices


def pl_list_logps(scaled_sims, idx):
    """Exact conditional-factorized log-probability of ordered lists under Plackett-Luce.

    log pi(D) = sum_i [ s_{d_i} - logsumexp_{d in remaining} s_d ]  — the denominator
    shrinks at each position (this is the term the top-N pool softmax lacks).
    scaled_sims: (B, M), differentiable; idx: (B, G, k). Returns (B, G).
    """
    B, G, k = idx.shape
    s = scaled_sims.unsqueeze(1).expand(B, G, scaled_sims.shape[-1])
    chosen = torch.zeros_like(s, dtype=torch.bool)
    lp = scaled_sims.new_zeros(B, G)
    for i in range(k):
        denom = torch.logsumexp(s.masked_fill(chosen, float("-inf")), dim=-1)
        s_i = s.gather(-1, idx[:, :, i : i + 1]).squeeze(-1)
        lp = lp + s_i - denom
        chosen = chosen.scatter(-1, idx[:, :, i : i + 1], True)
    return lp


def rloo_advantages(R):
    """Leave-one-out baseline over the group dim: A_g = R_g - mean_{h!=g} R_h. (B, G) -> (B, G).
    Unbiased and scale-preserving — no std division, so degenerate all-equal-reward groups give
    exactly zero advantage instead of noise amplified by a tiny denominator."""
    G = R.shape[1]
    return R - (R.sum(1, keepdim=True) - R) / max(G - 1, 1)


def infonce_loss(q, d, temperature=0.02):
    """In-batch contrastive anchor: positive = candidate 0 of each pool; negatives = every other
    doc in the batch. The absolute, cross-query constraint that prevents embedding collapse."""
    B, P, dim = d.shape
    flat = d.reshape(B * P, dim)
    logits = (q @ flat.T) / temperature
    labels = torch.arange(B, device=q.device) * P
    return F.cross_entropy(logits, labels)


def whiten(x, shift_mean=True):
    mean, var = x.mean(), x.var(unbiased=False)
    out = (x - mean) * torch.rsqrt(var + 1e-8)
    return out if shift_mean else out + mean


def compute_advantages_and_targets(rewards, old_values, old_logps, whiten_advantages=True):
    """rewards (B,P), old_values (B,), old_logps (B,P) -> (advantages (B,P), value_target (B,))."""
    value_target = (old_logps.exp() * rewards).sum(dim=-1)
    advantages = rewards - old_values.unsqueeze(1)
    if whiten_advantages:
        advantages = whiten(advantages)
    return advantages.detach(), value_target.detach()


def embedding_ppo_loss(
    logps, old_logps, advantages, values, old_values, value_target,
    epsilon_low=0.2, epsilon_high=0.2, cliprange_value=0.2, vf_coef=0.1,
    ref_logps=None, beta=0.0, disable_policy=False,
):
    ratio = torch.exp(logps - old_logps)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)
    pg_loss = torch.max(pg_losses1, pg_losses2).mean()

    if vf_coef == 0.0:
        vf_loss = logps.new_zeros(())
        vf_losses1 = vf_losses2 = None
    else:
        v_clipped = torch.clamp(values, old_values - cliprange_value, old_values + cliprange_value)
        vf_losses1 = (values - value_target) ** 2
        vf_losses2 = (v_clipped - value_target) ** 2
        vf_loss = 0.5 * torch.max(vf_losses1, vf_losses2).mean()

    kl = logps.new_zeros(())
    if beta != 0.0 and ref_logps is not None:
        kl = (logps.exp() * (logps - ref_logps)).sum(dim=-1).mean()

    loss = (vf_coef * vf_loss) if disable_policy else (pg_loss + vf_coef * vf_loss + beta * kl)

    with torch.no_grad():
        clipfrac = ((ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)).float().mean()
        metrics = {
            "loss/policy": pg_loss.item(),
            "loss/value": float(vf_loss),
            "loss/kl": float(kl),
            "ppo/ratio_mean": ratio.mean().item(),
            "ppo/clipfrac": clipfrac.item(),
            "ppo/value_mean": values.mean().item() if values is not None else 0.0,
            "ppo/advantage_mean": advantages.mean().item(),
        }
    return loss, metrics
