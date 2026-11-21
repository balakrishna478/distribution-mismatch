# Off-policy TD exact verification scripts

This repository provides reproducibility checks and exact peak certificates for the two-state off-policy linear TD(0) example.

## Requirements

- Python 3.9 or newer
- Dependencies listed in [requirements.txt](requirements.txt)

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Exact six-dimensional symmetric covariance certificate

Run from the repository root:

```bash
python code/verify_symmetric_covariance.py
```

This script constructs the covariance recursion directly on symmetric 2-by-2 blocks, using the coordinate order

`(Sigma_0[0,0], Sigma_0[1,1], Sigma_0[0,1], Sigma_1[0,0], Sigma_1[1,1], Sigma_1[0,1])`.

It uses exact rational arithmetic to verify:

1. The six-dimensional symmetric-block operator.
2. The factorization

   ```text
   det(I_6 - M_sym) =
   19701 eta^3 (1 - 50 eta)
   (990050 eta^2 - 29751 eta + 10000) / 250000000000.
   ```

3. The quadratic discriminant is negative and its leading coefficient is positive.
4. The stated positive-definite fixed covariance at `eta = 1/50` has exactly zero residual.
5. The determinant signs at representative step sizes below, at, and above the boundary.

## Full eight-dimensional lift and numerical checks

Run:

```bash
python code/verify_offpolicy_td.py
```

This retains the full-vectorization determinant check, boundary fixed-point check, characteristic polynomial/eigenvalue output, and finite-horizon peak calculations. The eight-dimensional determinant is not the determinant used in the six-dimensional symmetric-covariance stability theorem; use `verify_symmetric_covariance.py` for that certificate.

The finite-horizon peak calculations in `verify_offpolicy_td.py` are floating-point exploratory output; they are not the global-maximum proof. The proof is implemented separately below.

## Exact global-maximum certificates for Table 1

Run from the repository root:

```bash
bash code/certify_table1.sh
```

This runs `code/rigorous_peak.py`. For each of the ten step sizes in Table 1, it:

1. Builds the six-dimensional Markovian covariance operator using exact rational arithmetic.
2. Uses exact discrete-Lyapunov solutions to bound the error of fixed-point integer recursions.
3. Checks every iterate from zero through a finite cutoff, proving that none exceeds the tabulated peak.
4. Verifies a positive-semidefinite block certificate for the adjoint covariance recursion at that cutoff. Cone preservation then proves that the objective is nonincreasing for every later iteration.
5. Checks that the resulting rigorous interval rounds to the reported six-decimal peak.

The script writes `code/peak_certificates.json`. Its peak interval endpoints, finite-prefix gap, cone-margin bounds, Lyapunov traces, and rounding-error bounds are serialized as exact rational strings (numerator/denominator), not rounded decimal approximations. The certification uses exact integer/rational arithmetic; the initial eigenvector is handled exactly in its quadratic number field. A successful run ends with `PASS: all ten Table 1 entries are certified exact global maxima.`

When run locally, the JSON report remains at `code/peak_certificates.json`. In GitHub Actions, the workflow uploads it as the artifact `table1-peak-certificates`; download that artifact from the corresponding workflow run to inspect the machine-readable certificate report.

## Figures

The manuscript source package supplies the figure assets as static files. Figure-generation scripts are not included in this repository.

## Automated verification

GitHub Actions runs the two algebraic verification scripts and the Table 1 global-maximum certification on pushes and pull requests. The workflow fails if the JSON certificate report is not produced, and uploads that report as a workflow artifact.
