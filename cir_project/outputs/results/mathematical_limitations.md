## Mathematical Limitations of the CIR Framework

### 1. Single-Factor Constraint

The CIR model assumes all yields are driven by a single stochastic factor.
PCA on the training yield matrix reveals that PC1 explains
96.3% of variance, leaving 3.7%
unexplained. PC2 (3.0%) captures slope movements and
PC3 (0.5%) captures curvature -- both systematically missed
by any single-factor model. This means the CIR framework cannot reproduce
inverted yield curves, butterfly movements, or independent short/long-end
dynamics observed during Fed policy pivots.

### 2. Constant Parameter Assumption

Calibrating kappa, theta, sigma on the full sample implicitly assumes
time-homogeneous dynamics. Rolling calibration (Section 11) reveals the
most unstable parameter is **theta**
(CV=1.403).
The Feller condition is violated in a significant fraction of windows,
meaning the model's mathematical foundation (non-zero rates) breaks down
periodically. CIR++ addresses this via the deterministic shift phi(t),
but does not resolve the underlying parameter instability.

### 3. Gaussian Approximation in Discretization

Euler-Maruyama discretization replaces the exact non-central chi-squared
transition density with a Gaussian approximation. At the 1-day horizon,
the Wasserstein distance is 0.02 bps
(negligible), but at the 1-year horizon it grows to
0.61 bps.
For Monte Carlo pricing and risk simulation, this accumulation can bias
VaR and CVA calculations, particularly in the tails.

### 4. The Zero Lower Bound Problem

The CIR process is non-negative by construction (when Feller holds), but
cannot produce negative rates observed in EUR and JPY markets. The minimum
observed 3M yield in training data is 0.049%.
Under the calibrated model, P(r < 0.25% | 1Y) = 16.807%.
In a hypothetical low-rate environment (theta=0.1%, sigma=3%), the Feller
condition is violated,
and 59.6% of simulated paths pile up near zero.

### Implications for Extensions

- **CIR++** resolves the initial curve fitting problem via the shift but
  inherits all distributional and single-factor limitations.
- **CIR-J** adds jump risk but remains single-factor and constant-parameter.
- Neither extension addresses the multi-factor structure revealed by PCA.
- For production use, consider AFNS (Christensen et al. 2011) or HJM
  frameworks that naturally accommodate multiple factors.
