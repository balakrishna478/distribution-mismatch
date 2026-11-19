# Off-policy TD exact verification scripts

This repository provides reproducibility checks for the two-state off-policy linear TD(0) example.

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

All calculations in `verify_symmetric_covariance.py` are exact. The finite-horizon peak calculations in `verify_offpolicy_td.py` are floating-point numerical observations and are not global-maximum certificates.

## Automated verification

GitHub Actions runs both scripts on pushes and pull requests. The workflow checks that the exact algebraic assertions and numerical routines execute successfully; it does not convert finite-horizon peak observations into global-maximum certificates.
