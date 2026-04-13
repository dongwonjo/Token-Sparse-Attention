import torch
import triton
import triton.language as tl


# -----------------------------
# Kernel 1) compute per-query online-softmax stats over ALL keys
#   For each (b,h,q_in_window): compute
#     m = max_j score(q, k_j)
#     l = sum_j exp(score(q,k_j) - m)
#   with causal masking only for keys inside the window [L-W, L).
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({"BK": 64,  "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64,  "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["L", "D", "W"],
)
@triton.jit
def _stats_m_l_kernel(
    Q_ptr, K_ptr,
    M_ptr, L_ptr,                  # (B,H,W)
    B: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_ql: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kl: tl.constexpr, stride_kd: tl.constexpr,
    stride_mb: tl.constexpr, stride_mh: tl.constexpr, stride_mw: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lw: tl.constexpr,
    BK: tl.constexpr, BQ: tl.constexpr,
):
    pid_bh = tl.program_id(0)   # 0..B*H-1
    pid_qc = tl.program_id(1)   # query-chunk within W

    b = pid_bh // H
    h = pid_bh - b * H

    # window query indices [0..W)
    q_off = pid_qc * BQ + tl.arange(0, BQ)          # (BQ,)
    q_mask = q_off < W
    q_pos = (L - W) + q_off                         # absolute positions

    # load Q: (BQ, D) -> fp32
    d = tl.arange(0, D)
    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + q_pos[:, None] * stride_ql + d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    m = tl.full([BQ], -float("inf"), tl.float32)
    l_acc = tl.zeros([BQ], tl.float32)

    # scan keys in blocks
    k0 = 0
    inv_sqrt_d = 1.0 / tl.sqrt(tl.full([], D, tl.float32))
    win_start = L - W

    while k0 < L:
        k_idx = k0 + tl.arange(0, BK)              # (BK,)
        k_mask = k_idx < L

        # load K as (D, BK)
        k_ptrs = K_ptr + b * stride_kb + h * stride_kh + k_idx[None, :] * stride_kl + d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)

        # scores: (BQ, BK)
        scores = tl.dot(q, k) * inv_sqrt_d

        # causal only inside window keys:
        # if key in [L-W, L) and key > q_pos => -inf
        in_window_k = k_idx[None, :] >= win_start
        future = k_idx[None, :] > q_pos[:, None]
        scores = tl.where(in_window_k & future, -float("inf"), scores)

        # online softmax update
        row_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, row_max)
        l_acc = l_acc * tl.exp(m - m_new) + tl.sum(tl.exp(scores - m_new[:, None]), axis=1)
        m = m_new

        k0 += BK

    # store (B,H,W)
    m_ptrs = M_ptr + b * stride_mb + h * stride_mh + q_off * stride_mw
    l_ptrs = L_ptr + b * stride_lb + h * stride_lh + q_off * stride_lw
    tl.store(m_ptrs, m, mask=q_mask)
    tl.store(l_ptrs, l_acc, mask=q_mask)


# -----------------------------
# Kernel 2) compute prefix probs using stored (m,l)
#   out[b,h,k] = (1/W) * sum_{q in last W} exp(score(q,k)-m_q) / l_q
#   for k in [0, L-W)
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config({"BK": 64,  "BQ": 16, "num_warps": 4}, num_stages=3),
        triton.Config({"BK": 128, "BQ": 16, "num_warps": 8}, num_stages=3),
        triton.Config({"BK": 64,  "BQ": 32, "num_warps": 8}, num_stages=4),
    ],
    key=["L", "D", "W"],
)
@triton.jit
def _prefix_meanprob_kernel(
    Q_ptr, K_ptr,
    M_ptr, L_ptr,
    OUT_ptr,                        # (B,H,L-W)
    B: tl.constexpr, H: tl.constexpr,
    L: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
    stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_ql: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kl: tl.constexpr, stride_kd: tl.constexpr,
    stride_mb: tl.constexpr, stride_mh: tl.constexpr, stride_mw: tl.constexpr,
    stride_lb: tl.constexpr, stride_lh: tl.constexpr, stride_lw: tl.constexpr,
    stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_ok: tl.constexpr,
    BK: tl.constexpr, BQ: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_kb = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    prefix_len = L - W
    k_idx = pid_kb * BK + tl.arange(0, BK)
    k_mask = k_idx < prefix_len

    d = tl.arange(0, D)
    k_ptrs = K_ptr + b * stride_kb + h * stride_kh + k_idx[None, :] * stride_kl + d[:, None] * stride_kd
    k = tl.load(k_ptrs, mask=k_mask[None, :], other=0.0).to(tl.float32)  # (D, BK)

    acc = tl.zeros([BK], tl.float32)
    inv_sqrt_d = 1.0 / tl.sqrt(tl.full([], D, tl.float32))

    q0 = 0
    while q0 < W:
        q_off = q0 + tl.arange(0, BQ)
        q_mask = q_off < W
        q_pos = (L - W) + q_off

        q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + q_pos[:, None] * stride_ql + d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)  # (BQ, D)

        m_ptrs = M_ptr + b * stride_mb + h * stride_mh + q_off * stride_mw
        l_ptrs = L_ptr + b * stride_lb + h * stride_lh + q_off * stride_lw
        m = tl.load(m_ptrs, mask=q_mask, other=-float("inf")).to(tl.float32)
        l = tl.load(l_ptrs, mask=q_mask, other=1.0).to(tl.float32)

        scores = tl.dot(q, k) * inv_sqrt_d  # (BQ, BK)

        # prefix keys는 항상 q_pos보다 과거이므로 causal 마스킹 불필요
        probs = tl.exp(scores - m[:, None]) / l[:, None]  # (BQ, BK)
        acc += tl.sum(probs, axis=0)

        q0 += BQ

    acc *= 1.0 / tl.full([], W, tl.float32)

    out_ptrs = OUT_ptr + b * stride_ob + h * stride_oh + k_idx * stride_ok
    tl.store(out_ptrs, acc, mask=k_mask)


# -----------------------------
# Python wrapper
# -----------------------------
def triton_get_attn_cache(
    query_states: torch.Tensor,  # (B,H,L,D)
    key_states: torch.Tensor,    # (B,H,L,D)  <-- already repeated
    window_size: int,
    out_dtype=torch.float32,
) -> torch.Tensor:
    """
    Returns attn_cache = mean_q softmax(Q_window @ K^T)[prefix_keys]
    Shape: (B,H,L-W), dtype fp32
    """
    assert query_states.is_cuda and key_states.is_cuda
    assert query_states.ndim == 4 and key_states.ndim == 4
    B, H, L, D = query_states.shape
    B2, H2, L2, D2 = key_states.shape
    assert (B, H, L, D) == (B2, H2, L2, D2)

    W = window_size
    assert 0 < W <= L
    prefix_len = L - W
    if prefix_len <= 0:
        return query_states.new_zeros((B, H, 0), dtype=torch.float32)

    # (B,H,W)
    m = torch.empty((B, H, W), device=query_states.device, dtype=torch.float32)
    l = torch.empty((B, H, W), device=query_states.device, dtype=torch.float32)

    # (B,H,L-W)
    out = torch.empty((B, H, prefix_len), device=query_states.device, dtype=torch.float32)

    # grid: (B*H, ceil(W/BQ)) - we set 2D grid, BQ autotuned; use safe upper bound with 16
    grid_stats = (B * H, triton.cdiv(W, 16))
    _stats_m_l_kernel[grid_stats](
        query_states, key_states, m, l,
        B=B, H=H, L=L, D=D, W=W,
        stride_qb=query_states.stride(0),
        stride_qh=query_states.stride(1),
        stride_ql=query_states.stride(2),
        stride_qd=query_states.stride(3),
        stride_kb=key_states.stride(0),
        stride_kh=key_states.stride(1),
        stride_kl=key_states.stride(2),
        stride_kd=key_states.stride(3),
        stride_mb=m.stride(0),
        stride_mh=m.stride(1),
        stride_mw=m.stride(2),
        stride_lb=l.stride(0),
        stride_lh=l.stride(1),
        stride_lw=l.stride(2),
    )

    # grid: (B*H, ceil(prefix_len/BK)) - safe bound with 64
    grid_prob = (B * H, triton.cdiv(prefix_len, 64))
    _prefix_meanprob_kernel[grid_prob](
        query_states, key_states, m, l, out,
        B=B, H=H, L=L, D=D, W=W,
        stride_qb=query_states.stride(0),
        stride_qh=query_states.stride(1),
        stride_ql=query_states.stride(2),
        stride_qd=query_states.stride(3),
        stride_kb=key_states.stride(0),
        stride_kh=key_states.stride(1),
        stride_kl=key_states.stride(2),
        stride_kd=key_states.stride(3),
        stride_mb=m.stride(0),
        stride_mh=m.stride(1),
        stride_mw=m.stride(2),
        stride_lb=l.stride(0),
        stride_lh=l.stride(1),
        stride_lw=l.stride(2),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ok=out.stride(2),
    )
    return out.to(out_dtype)