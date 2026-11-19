#!/usr/bin/env python3
"""
Reproducibility script for the two-state off-policy linear-TD example.

Requires: numpy, sympy
Run: python verify_offpolicy_td.py

The script:
  (1) constructs the exact 8 x 8 Markovian covariance lift,
  (2) verifies the determinant factorization,
  (3) verifies the positive-definite fixed covariance at eta=0.02,
  (4) computes the boundary characteristic polynomial and eigenvalues,
  (5) estimates finite-horizon transient peaks for selected step sizes.

Vectorization convention: column-major (Fortran) order.
State-conditioned covariance convention:
    Sigma_k[i] = E[theta_k theta_k^T 1{S_k=i}].
"""
from fractions import Fraction
import sympy as sp
import numpy as np

gamma = sp.Rational(99, 100)
p_target = sp.Rational(1, 100)
p_alt = sp.Rational(99, 100)

# Transition probabilities P(j | i, a), with states/actions indexed 0,1.
P_target = sp.Matrix([[0, 1], [1, 0]])
P_alt = sp.Matrix([[1, 0], [sp.Rational(98, 99), sp.Rational(1, 99)]])

# Behavior stationary distribution.
d = [sp.Rational(99, 100), sp.Rational(1, 100)]

I2 = sp.eye(2)
E = [sp.Matrix([1, 0]), sp.Matrix([0, 1])]

def sample_update_matrix(i, j, action, eta):
    """B = I - eta A(i,a,j); target-action IS ratio is 100, alt ratio is 0."""
    rho = sp.Integer(100) if action == 0 else sp.Integer(0)
    A_sample = rho * E[i] * (E[i] - gamma * E[j]).T
    return I2 - eta * A_sample

def lifted_matrix(eta):
    """Return exact 8x8 M_eta for stacked [vec(Sigma_0); vec(Sigma_1)]."""
    M = sp.zeros(8, 8)
    transitions = [
        (0, 0, p_target, P_target),
        (0, 1, p_alt, P_alt),
        (1, 0, p_target, P_target),
        (1, 1, p_alt, P_alt),
    ]
    for i, action, action_prob, P in transitions:
        for j in range(2):
            prob = action_prob * P[i, j]
            if prob == 0:
                continue
            B = sample_update_matrix(i, j, action, eta)
            # vec(B Sigma B^T) = (B kron B) vec(Sigma), column-major.
            block = prob * sp.kronecker_product(B, B)
            M[4*j:4*(j+1), 4*i:4*(i+1)] += block
    return M

eta = sp.symbols("eta", real=True)
M = lifted_matrix(eta)

print("Computing exact determinant factorization...")
det_factor = sp.factor((sp.eye(8) - M).det())
expected = (
    sp.Integer(19701) * eta**4 * (eta - 1) * (50*eta - 1)
    * (990050*eta**2 - 29751*eta + 10000)
    / sp.Integer(250000000000)
)
assert sp.simplify(det_factor - expected) == 0
print("det(I - M_eta) factorization verified:")
print(det_factor)

# Verify the positive-definite fixed covariance at eta = 1/50.
eta_star = sp.Rational(1, 50)
M_star = lifted_matrix(eta_star)
Sigma0 = sp.Matrix([[99, sp.Rational(9801, 100)],
                    [sp.Rational(9801, 100), 99]])
Sigma1 = sp.Matrix([[1, sp.Rational(99, 100)],
                    [sp.Rational(99, 100), 1]])
z_star = sp.Matrix(list(Sigma0) + list(Sigma1))
residual = sp.simplify(M_star * z_star - z_star)
assert residual == sp.zeros(8, 1)
assert min(sp.Matrix(Sigma0).eigenvals().keys()) > 0
assert min(sp.Matrix(Sigma1).eigenvals().keys()) > 0
print("\nBoundary fixed covariance verified; residual:")
print(residual)
print("eig(Sigma0) =", Sigma0.eigenvals())
print("eig(Sigma1) =", Sigma1.eigenvals())

# Boundary characteristic polynomial and numerical eigenvalues.
lam = sp.symbols("lambda")
charpoly = sp.factor((lam * sp.eye(8) - M_star).det())
print("\nCharacteristic polynomial at eta=1/50:")
print(charpoly)
eigvals = np.linalg.eigvals(np.array(M_star.evalf(), dtype=float))
print("Numerical eigenvalues at eta=0.02:")
for z in sorted(eigvals, key=lambda x: -abs(x)):
    print(f"  {z:.12g}")

# The symmetric-part negative eigenvector from the mean update example.
v = np.array([-0.38502705, -0.92290529], dtype=float)
v /= np.linalg.norm(v)
d_float = np.array([0.99, 0.01])

def lift_float(step):
    """Floating-point version of M_eta, using the same block convention."""
    Mv = np.zeros((8, 8), dtype=float)
    Ps = [np.array([[0., 1.], [1., 0.]]),
          np.array([[1., 0.], [98./99., 1./99.]])]
    action_probs = [0.01, 0.99]
    ratios = [100., 0.]
    for i in range(2):
        for a in range(2):
            for j in range(2):
                prob = action_probs[a] * Ps[a][i, j]
                if prob == 0:
                    continue
                ei = np.eye(2)[:, i]
                ej = np.eye(2)[:, j]
                A = ratios[a] * np.outer(ei, ei - 0.99*ej)
                B = np.eye(2) - step*A
                Mv[4*j:4*(j+1), 4*i:4*(i+1)] += prob*np.kron(B, B)
    return Mv

def finite_horizon_peak(step, max_iter):
    """Iterate exact Markovian covariance recursion; return observed peak."""
    Mv = lift_float(step)
    # vec(v v^T) in Fortran order, weighted by initial stationary state mass.
    vv = np.outer(v, v)
    z = np.concatenate([0.99*vv.reshape(-1, order="F"),
                        0.01*vv.reshape(-1, order="F")])
    peak = float(np.trace(vv))
    peak_k = 0
    for k in range(1, max_iter + 1):
        z = Mv @ z
        S0 = z[:4].reshape((2, 2), order="F")
        S1 = z[4:].reshape((2, 2), order="F")
        value = float(np.trace(S0) + np.trace(S1))
        if value > peak:
            peak, peak_k = value, k
    return peak, peak_k

print("\nFinite-horizon observed peaks (not certified global maxima):")
peak_settings = [
    (0.00100, 1_000_000),
    (0.00500, 300_000),
    (0.01000, 200_000),
    (0.01500, 100_000),
    (0.01900, 100_000),
    (0.01950, 100_000),
    (0.01980, 300_000),
    (0.01990, 300_000),
    (0.01995, 300_000),
    (0.01999, 300_000),
]
for step, horizon in peak_settings:
    peak, k_peak = finite_horizon_peak(step, horizon)
    print(f"eta={step:.5f}  horizon={horizon:>7d}  "
          f"peak={peak:.9f}  k_peak={k_peak}")

print("\nAll exact algebraic checks passed.")
