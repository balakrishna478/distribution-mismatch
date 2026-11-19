#!/usr/bin/env python3
"""Exact six-dimensional symmetric covariance-lift certificate.

Run: python verify_symmetric_covariance.py
Requires: sympy

The state-conditioned covariance blocks are symmetric 2x2 matrices
Sigma_i = [[x_i, z_i], [z_i, y_i]].  The stacked coordinate order is
(x_0, y_0, z_0, x_1, y_1, z_1).  All arithmetic is exact rational arithmetic.
"""
import sympy as sp

eta = sp.symbols("eta", real=True)
gamma = sp.Rational(99, 100)

# Transition matrices: action 0 is the target action; action 1 is alternative.
P_target = sp.Matrix([[0, 1], [1, 0]])
P_alt = sp.Matrix([[1, 0], [sp.Rational(98, 99), sp.Rational(1, 99)]])
action_prob = (sp.Rational(1, 100), sp.Rational(99, 100))
importance_ratio = (sp.Integer(100), sp.Integer(0))
e = (sp.Matrix([1, 0]), sp.Matrix([0, 1]))

# Symmetric coordinate basis: (Sigma[0,0], Sigma[1,1], Sigma[0,1]).
def from_coords(x, y, z):
    return sp.Matrix([[x, z], [z, y]])

basis = (
    sp.Matrix([[1, 0], [0, 0]]),
    sp.Matrix([[0, 0], [0, 1]]),
    sp.Matrix([[0, 1], [1, 0]]),
)

def to_coords(X):
    assert X == X.T
    return sp.Matrix([X[0, 0], X[1, 1], X[0, 1]])

def sample_B(i, j, a):
    A = importance_ratio[a] * e[i] * (e[i] - gamma * e[j]).T
    return sp.eye(2) - eta * A

# Construct M_sym by applying the actual covariance recursion to each of
# the six coordinate basis vectors. This avoids any full-vectorization
# reduction or assumptions about off-diagonal coordinates.
M_sym = sp.zeros(6, 6)
for source_state in range(2):
    for q, Xbasis in enumerate(basis):
        input_blocks = [sp.zeros(2), sp.zeros(2)]
        input_blocks[source_state] = Xbasis
        output_blocks = [sp.zeros(2), sp.zeros(2)]
        for i in range(2):
            for a, P in enumerate((P_target, P_alt)):
                for j in range(2):
                    probability = action_prob[a] * P[i, j]
                    if probability:
                        B = sample_B(i, j, a)
                        output_blocks[j] += probability * B * input_blocks[i] * B.T
        col = 3 * source_state + q
        for j in range(2):
            M_sym[3*j:3*j+3, col] = to_coords(output_blocks[j])

print("Exact 6x6 symmetric covariance operator:")
sp.pprint(M_sym)

actual_det = sp.factor((sp.eye(6) - M_sym).det())
expected_det = (
    sp.Integer(19701) * eta**3 * (1 - 50*eta)
    * (990050*eta**2 - 29751*eta + 10000)
    / sp.Integer(250000000000)
)
assert sp.factor(actual_det - expected_det) == 0
print("\ndet(I_6 - M_sym) factorization verified:")
sp.pprint(actual_det)

quadratic = 990050*eta**2 - 29751*eta + 10000
discriminant = sp.discriminant(quadratic, eta)
assert discriminant < 0
assert sp.LC(sp.Poly(quadratic, eta)) > 0
print("\nQuadratic discriminant:", discriminant)
print("Quadratic is strictly positive on R (negative discriminant, positive leading coefficient).")

# Check the exact fixed covariance at eta*=1/50.
eta_star = sp.Rational(1, 50)
M_star = M_sym.subs(eta, eta_star)
Sigma0 = sp.Matrix([[99, sp.Rational(9801, 100)],
                    [sp.Rational(9801, 100), 99]])
Sigma1 = sp.Matrix([[1, sp.Rational(99, 100)],
                    [sp.Rational(99, 100), 1]])
z_star = sp.Matrix([
    Sigma0[0, 0], Sigma0[1, 1], Sigma0[0, 1],
    Sigma1[0, 0], Sigma1[1, 1], Sigma1[0, 1],
])
residual = sp.simplify(M_star*z_star - z_star)
assert residual == sp.zeros(6, 1)
assert sp.factor(Sigma0.det()) > 0 and Sigma0[0, 0] > 0
assert sp.factor(Sigma1.det()) > 0 and Sigma1[0, 0] > 0
print("\nBoundary fixed point residual:")
sp.pprint(residual)
print("eig(Sigma_0):", Sigma0.eigenvals())
print("eig(Sigma_1):", Sigma1.eigenvals())

# Verify determinant sign on either side of the claimed positive boundary.
for value, sign in ((sp.Rational(1, 100), 1),
                    (sp.Rational(1, 50), 0),
                    (sp.Rational(3, 100), -1)):
    det_value = sp.factor(actual_det.subs(eta, value))
    assert (det_value > 0) if sign == 1 else ((det_value == 0) if sign == 0 else (det_value < 0))
    print(f"det(I-M_sym) at eta={value}: {det_value}")

print("\nPASS: all exact six-dimensional symmetric-lift checks passed.")
