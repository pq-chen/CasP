import torch
from torch import Tensor

def _ensure_matrix_rhs(b: Tensor) -> Tensor:
    # Accept (..., N) or (..., N, 1)
    if b.dim() >= 2 and b.shape[-1] == 1:
        return b
    return b.unsqueeze(-1)

def _restore_rhs_shape(x: Tensor, b_shape) -> Tensor:
    # If original b was (..., N), squeeze the last dim back
    return x.squeeze(-1) if (len(b_shape) == x.dim() and b_shape[-1] != 1) else x

def solve2x2(A: Tensor, b: Tensor, eps: float = 1e-9) -> Tensor:
    # A: (..., 2, 2), b: (..., 2) or (..., 2, 1)
    b_in = b
    b = _ensure_matrix_rhs(b)
    a, c = A[..., 0, 0], A[..., 1, 0]
    b_, d = A[..., 0, 1], A[..., 1, 1]
    det = a * d - b_ * c
    det = det.clamp_min(eps)
    invA = torch.empty_like(A)
    invA[..., 0, 0] =  d / det
    invA[..., 0, 1] = -b_ / det
    invA[..., 1, 0] = -c / det
    invA[..., 1, 1] =  a / det
    x = invA.matmul(b)
    return _restore_rhs_shape(x, b_in.shape)

def solve3x3(A: Tensor, b: Tensor, eps: float = 1e-9) -> Tensor:
    # A: (..., 3, 3), b: (..., 3) or (..., 3, 1)
    b_in = b
    b = _ensure_matrix_rhs(b)
    a11, a12, a13 = A[...,0,0], A[...,0,1], A[...,0,2]
    a21, a22, a23 = A[...,1,0], A[...,1,1], A[...,1,2]
    a31, a32, a33 = A[...,2,0], A[...,2,1], A[...,2,2]
    # Cofactors
    c11 =  a22*a33 - a23*a32
    c12 = -(a21*a33 - a23*a31)
    c13 =  a21*a32 - a22*a31
    c21 = -(a12*a33 - a13*a32)
    c22 =  a11*a33 - a13*a31
    c23 = -(a11*a32 - a12*a31)
    c31 =  a12*a23 - a13*a22
    c32 = -(a11*a23 - a13*a21)
    c33 =  a11*a22 - a12*a21
    det = a11*c11 + a12*c12 + a13*c13
    det = det.clamp_min(eps)
    # inv(A) = adj(A)/det ; adj(A) = cof(A)^T
    invA = torch.empty_like(A)
    invA[...,0,0] = c11 / det; invA[...,0,1] = c21 / det; invA[...,0,2] = c31 / det
    invA[...,1,0] = c12 / det; invA[...,1,1] = c22 / det; invA[...,1,2] = c32 / det
    invA[...,2,0] = c13 / det; invA[...,2,1] = c23 / det; invA[...,2,2] = c33 / det
    x = invA.matmul(b)
    return _restore_rhs_shape(x, b_in.shape)

@torch.no_grad()
def cg_solve_spd(A: Tensor, b: Tensor, iters: int = 8, tol: float = 0.0) -> Tensor:
    """
    Conjugate Gradient for SPD A using only matmul/add/mul (ONNX friendly).
    A: (..., N, N), b: (..., N) or (..., N, 1)
    """
    b_in = b
    b = _ensure_matrix_rhs(b)
    *batch, N, _ = A.shape
    x = torch.zeros((*batch, N, 1), dtype=A.dtype, device=A.device)
    r = b - A.matmul(x)
    p = r.clone()
    rs_old = (r.transpose(-2, -1).matmul(r)).squeeze(-1).squeeze(-1)  # (...,)

    for _ in range(iters):
        Ap = A.matmul(p)
        pAp = (p.transpose(-2, -1).matmul(Ap)).squeeze(-1).squeeze(-1) + 1e-12
        alpha = (rs_old / pAp).unsqueeze(-1).unsqueeze(-1)            # (...,1,1)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r.transpose(-2, -1).matmul(r)).squeeze(-1).squeeze(-1)
        if tol > 0.0:
            # Keep it tensor-only (no Python branching per element)
            # We still iterate fixed `iters` to keep graph static.
            pass
        beta = (rs_new / (rs_old + 1e-12)).unsqueeze(-1).unsqueeze(-1)
        p = r + beta * p
        rs_old = rs_new

    return _restore_rhs_shape(x, b_in.shape)

def onnx_safe_solve(A: Tensor, b: Tensor, assume_spd: bool = True) -> Tensor:
    """
    Dispatch to ONNX-safe solvers:
      - 2x2 / 3x3 analytic
      - else CG for SPD (matmul-only)
      - if not SPD, you can switch assume_spd=False and we’ll use GD steps.
    """
    n = A.shape[-1]
    if n == 2:
        return solve2x2(A, b)
    if n == 3:
        return solve3x3(A, b)
    if assume_spd:
        return cg_solve_spd(A, b, iters=8)
    # Gradient descent fallback (non-SPD): x_{k+1} = x_k - α A^T(Ax - b)
    b_in = b
    b = _ensure_matrix_rhs(b)
    x = torch.zeros((*A.shape[:-1], 1), dtype=A.dtype, device=A.device)  # (..., N, 1)
    # constant step via Lipschitz upper bound heuristics (no eigendecomp)
    # α = 1 / (trace(A^T A)/N + ε)
    ATA_trace = (A.transpose(-2, -1).matmul(A)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (...,)
    alpha = (1.0 / (ATA_trace / A.shape[-1] + 1e-6)).unsqueeze(-1).unsqueeze(-1)
    for _ in range(16):
        Ax = A.matmul(x)
        grad = A.transpose(-2, -1).matmul(Ax - b)
        x = x - alpha * grad
    return _restore_rhs_shape(x, b_in.shape)
