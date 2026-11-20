#!/usr/bin/env python3
"""Exact global-maximum certificates for all ten Table 1 covariance peaks.

The proof has two parts for each eta:
  (1) exact fixed-point integer arithmetic plus a discrete-Lyapunov error bound
      verifies every iterate k=0,...,K0 is below the tabulated iterate;
  (2) an exact cone certificate Q_K0 - M* Q_K0 >= 0 proves monotonic decay
      for every k >= K0, since the covariance adjoint preserves the PSD cone.

No floating-point iteration is used in the certification. SymPy rational
arithmetic constructs the exact operators and Lyapunov matrices. The finite
recursions use integer arithmetic on a 2^BITS grid, with an explicit rigorous
rounding-error bound derived from the Lyapunov matrices.
"""
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp

ETAS = ["0.00100", "0.00500", "0.01000", "0.01500", "0.01900",
        "0.01950", "0.01980", "0.01990", "0.01995", "0.01999"]
PEAK_K = [7269, 1450, 691, 255, 225, 282, 4458, 10324, 19600, 61544]
K_MONO = [11071, 2127, 712, 347, 271, 297, 8130, 15125, 25437, 68814]
PEAK_TEXT = ["1.664493", "1.672292", "1.681504", "1.717240", "1.895948",
             "1.951277", "2.110078", "2.468602", "3.104982", "6.145016"]
BITS = 512
SCALE = 1 << BITS

GAMMA = sp.Rational(99, 100)
P_TARGET = sp.Matrix([[0, 1], [1, 0]])
P_ALT = sp.Matrix([[1, 0], [sp.Rational(98, 99), sp.Rational(1, 99)]])
ACTION_PROB = (sp.Rational(1, 100), sp.Rational(99, 100))
IS_RATIO = (sp.Integer(100), sp.Integer(0))
E = (sp.Matrix([1, 0]), sp.Matrix([0, 1]))
SYM_BASIS = (sp.Matrix([[1, 0], [0, 0]]),
             sp.Matrix([[0, 0], [0, 1]]),
             sp.Matrix([[0, 1], [1, 0]]))
WEIGHT = sp.diag(1, 1, 2, 1, 1, 2)


def build_covariance_operator(eta: sp.Rational) -> sp.Matrix:
    """Exact 6x6 operator in coordinates (x0,y0,z0,x1,y1,z1)."""
    M = sp.zeros(6)
    for i in range(2):
        for q, Xbasis in enumerate(SYM_BASIS):
            inputs = [sp.zeros(2), sp.zeros(2)]
            inputs[i] = Xbasis
            outputs = [sp.zeros(2), sp.zeros(2)]
            for action, transition in enumerate((P_TARGET, P_ALT)):
                for j in range(2):
                    probability = ACTION_PROB[action] * transition[i, j]
                    if probability == 0:
                        continue
                    A_sample = IS_RATIO[action] * E[i] * (E[i] - GAMMA * E[j]).T
                    B = sp.eye(2) - eta * A_sample
                    outputs[j] += probability * B * inputs[i] * B.T
            for j in range(2):
                X = outputs[j]
                M[3*j:3*j+3, 3*i+q] = sp.Matrix([X[0, 0], X[1, 1], X[0, 1]])
    return M


def exact_initial_covariance() -> sp.Matrix:
    """Stationary-state initial covariance for theta_0=v, exact in Q(sqrt(19405))."""
    ratio = sp.Rational(98, 99) + sp.sqrt(19405) / 99
    v1 = -1 / sp.sqrt(1 + ratio**2)
    v2 = -ratio / sp.sqrt(1 + ratio**2)
    vv = sp.Matrix([[v1*v1, v1*v2], [v1*v2, v2*v2]])
    S0 = sp.Rational(99, 100) * vv
    S1 = sp.Rational(1, 100) * vv
    return sp.Matrix([S0[0, 0], S0[1, 1], S0[0, 1],
                      S1[0, 0], S1[1, 1], S1[0, 1]])


def sign_in_quadratic_field(expr: sp.Expr) -> int:
    """Exact sign of a+b*sqrt(19405), with rational a,b."""
    root = sp.sqrt(19405)
    reduced = sp.radsimp(sp.simplify(expr))
    poly = sp.Poly(sp.expand(reduced), root)
    if poly.degree() > 1:
        raise ArithmeticError("initial value left the quadratic field")
    coeff = poly.all_coeffs()
    if poly.degree() == 0:
        a, b = coeff[0], sp.Rational(0)
    else:
        b, a = coeff
    a, b = sp.Rational(a), sp.Rational(b)
    if b == 0:
        return 1 if a > 0 else (-1 if a < 0 else 0)
    if a == 0:
        return 1 if b > 0 else -1
    if a > 0 and b > 0:
        return 1
    if a < 0 and b < 0:
        return -1
    cmp = sp.sign(a*a - b*b*19405)
    return int(cmp) if a > 0 else -int(cmp)


def round_algebraic_to_grid(x: sp.Expr) -> int:
    """Nearest integer to SCALE*x, with an exact quadratic-field enclosure."""
    approx = sp.N(x * SCALE, 500)
    q = int(sp.floor(approx + sp.Rational(1, 2)))
    lo = sp.Rational(2*q - 1, 2*SCALE)
    hi = sp.Rational(2*q + 1, 2*SCALE)
    if not (sign_in_quadratic_field(x - lo) > 0 and sign_in_quadratic_field(hi - x) > 0):
        raise ArithmeticError("could not certify initial algebraic rounding")
    return q


def row_integer_data(A: sp.Matrix):
    """Represent each rational row with one common positive denominator."""
    rows = []
    for i in range(A.rows):
        den = sp.ilcm(*[sp.denom(A[i, j]) for j in range(A.cols)])
        coeff = [int(A[i, j] * den) for j in range(A.cols)]
        rows.append((int(den), coeff))
    return rows


def nearest_integer_ratio(num: int, den: int) -> int:
    """Exact nearest integer to num/den (ties away from zero)."""
    if den <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if num < 0 else 1
    a = abs(num)
    q, r = divmod(a, den)
    if 2*r >= den:
        q += 1
    return sign*q


def fixed_step(rows, z: list[int]) -> list[int]:
    """One exact integer-grid step; each coordinate incurs <= 1/(2*SCALE)."""
    out = []
    for den, coeff in rows:
        num = sum(c*x for c, x in zip(coeff, z))
        out.append(nearest_integer_ratio(num, den))
    return out


def discrete_lyapunov(A: sp.Matrix) -> tuple[sp.Matrix, sp.Rational]:
    """Solve P-A^T P A=I exactly and return P, trace(P)."""
    n = A.rows
    K = sp.eye(n*n) - sp.kronecker_product(A.T, A.T)
    rhs = sp.Matrix([1 if i % (n+1) == 0 else 0 for i in range(n*n)])
    vecP = K.LUsolve(rhs)
    P = sp.Matrix(n, n, lambda i, j: vecP[i + j*n])
    if P != P.T or P - A.T*P*A != sp.eye(n):
        raise ArithmeticError("exact Lyapunov equation failed")
    # Sylvester's criterion gives exact positive definiteness.
    for k in range(1, n+1):
        if P[:k, :k].det() <= 0:
            raise ArithmeticError("Lyapunov matrix is not positive definite")
    trace = sp.Rational(P.trace())
    if trace < 1:
        raise ArithmeticError("unexpected Lyapunov trace")
    return P, trace


def rational_round6(x: sp.Rational) -> str:
    q = sp.floor(x * 10**6 + sp.Rational(1, 2))
    return f"{int(q)//10**6}.{int(q)%10**6:06d}"


def certify_one(eta_text: str, peak_k: int, K0: int, expected: str, x0: sp.Matrix):
    eta = sp.Rational(eta_text)
    M = build_covariance_operator(eta)
    Madj = WEIGHT.inv() * M.T * WEIGHT

    # Exact discrete Lyapunov matrices provide uniform error bounds for the
    # fixed-point recursions of Sigma_k and Q_k=(M*)^k I.
    _, trace_M = discrete_lyapunov(M)
    _, trace_A = discrete_lyapunov(Madj)
    err_state = sp.Rational(4) * trace_M**2 / SCALE
    err_dual = sp.Rational(3) * trace_A**2 / SCALE
    max_abs_A = max(abs(Madj[i, j]) for i in range(6) for j in range(6))
    err_D = (1 + 6*max_abs_A) * err_dual

    M_rows = row_integer_data(M)
    A_rows = row_integer_data(Madj)

    # Forward covariance sequence: exact integer-grid recursion with a
    # rigorously bounded Euclidean error <= err_state at every k.
    x = [round_algebraic_to_grid(x0[i]) for i in range(6)]
    peak_hat = None
    max_other_upper = None
    for k in range(K0 + 1):
        f_hat = sp.Rational(x[0] + x[1] + x[3] + x[4], SCALE)
        if k == peak_k:
            peak_hat = f_hat
        else:
            upper = f_hat + 2*err_state
            if max_other_upper is None or upper > max_other_upper:
                max_other_upper = upper
        if k < K0:
            x = fixed_step(M_rows, x)
    if peak_hat is None:
        raise ArithmeticError("peak iteration not reached")
    peak_lower = peak_hat - 2*err_state
    peak_upper = peak_hat + 2*err_state
    if max_other_upper >= peak_lower:
        raise ArithmeticError(f"finite prefix not strictly below peak at eta={eta_text}")
    if rational_round6(peak_lower) != expected or rational_round6(peak_upper) != expected:
        raise ArithmeticError(f"peak interval does not round to {expected} at eta={eta_text}")

    # Dual sequence Q_k=(M*)^k I. At K0, certify D=Q_K0-Q_(K0+1)
    # positive definite blockwise. Since M* preserves the PSD cone,
    # Q_k-Q_(k+1)=(M*)^(k-K0)D >= 0 for every k>=K0.
    q = [SCALE if i in (0, 1, 3, 4) else 0 for i in range(6)]
    for _ in range(K0):
        q = fixed_step(A_rows, q)
    Dhat = []
    for i in range(6):
        exact_row = sum(Madj[i, j] * q[j] for j in range(6))
        Dhat.append(sp.Rational(q[i], SCALE) - exact_row / SCALE)

    cone_margins = []
    for b in range(2):
        a = Dhat[3*b]
        d = Dhat[3*b+1]
        off = Dhat[3*b+2]
        a_lo = a - err_D
        d_lo = d - err_D
        off_abs_hi = abs(off) + err_D
        det_lo = a_lo*d_lo - off_abs_hi**2
        if not (a_lo > 0 and d_lo > 0 and det_lo > 0):
            raise ArithmeticError(f"dual cone certificate failed at eta={eta_text}, block={b}")
        cone_margins.append({"a_lower": str(a_lo), "d_lower": str(d_lo),
                             "determinant_lower": str(det_lo)})

    return {
        "eta": eta_text,
        "peak_iteration": peak_k,
        "monotonicity_from_iteration": K0,
        "peak_interval": [str(sp.N(peak_lower, 40)), str(sp.N(peak_upper, 40))],
        "peak_rounded": expected,
        "finite_prefix_gap_lower": str(sp.N(peak_lower - max_other_upper, 30)),
        "dual_cone_margins": [
            {k: str(sp.N(sp.Rational(v), 25)) for k, v in item.items()}
            for item in cone_margins
        ],
        "state_lyapunov_trace": str(sp.N(trace_M, 25)),
        "dual_lyapunov_trace": str(sp.N(trace_A, 25)),
        "state_error_bound": str(sp.N(err_state, 12)),
        "dual_error_bound": str(sp.N(err_dual, 12)),
        "dual_D_entry_error_bound": str(sp.N(err_D, 12)),
    }


def main():
    x0 = exact_initial_covariance()
    results = []
    for eta, kp, K0, peak in zip(ETAS, PEAK_K, K_MONO, PEAK_TEXT):
        result = certify_one(eta, kp, K0, peak, x0)
        results.append(result)
        print(f"PASS eta={eta}: peak at k={kp}, exact prefix checked through k={K0}; "
              f"monotone thereafter; peak interval rounds to {peak}.")
        print("  finite-prefix gap lower bound:", result["finite_prefix_gap_lower"])
        print("  cone determinant lower bounds:", [m["determinant_lower"] for m in result["dual_cone_margins"]])
    Path(__file__).with_name("peak_certificates.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PASS: all ten Table 1 entries are certified exact global maxima.")
    print("Wrote peak_certificates.json")


if __name__ == "__main__":
    main()
