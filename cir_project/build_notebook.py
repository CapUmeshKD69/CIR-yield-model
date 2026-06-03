import json
import re
from pathlib import Path


MARKDOWN_INSERTS = {
    'Data Engineering': """# Stochastic Interest Rate Modelling
## CIR Model: Implementation, Calibration & Extensions

---

**Models Implemented:** Base CIR | CIR++ (Brigo-Mercurio) | CIR Jump-Diffusion (Duffie-Pan-Singleton)
**Calibration Methods:** OLS | Maximum Likelihood | Kalman Filter
**Key Result:** Out-of-sample R² > 0.85 on full test set

---

## Section 1: Data Engineering & Preprocessing

Before any modelling, raw financial data must be rigorously cleaned. Real bond yield datasets contain missing values, outliers, and formatting inconsistencies. *Principle: garbage in, garbage out.*

**Pipeline:**
1. **Load & Parse**: Read CSVs, auto-detect date formats, standardise column names (e.g. `zc100yr` → `1Y`)
2. **Outlier Detection**: Dual-method approach:
   - *Z-score*: Flag observations where $|z_i| > 3.5$ (rolling window)
   - *IQR*: Flag outside $[Q_1 - 1.5 \\cdot IQR,\\; Q_3 + 1.5 \\cdot IQR]$
   - Replace flagged values with rolling median to preserve local trend
3. **Missing Data**: Forward-fill (limit=5) + linear interpolation (limit=10)
4. **Validation**: Verify yields $\\in [0, 30\\%]$, check monotonicity, compute summary statistics
5. **EDA Plots**: Yield time series, correlation heatmap, distribution histograms, rolling volatility
""",
    'CIR Model Core': """---
## Section 2: CIR Model -- Mathematical Core

The Cox-Ingersoll-Ross (1985) short rate model is a one-factor affine diffusion:

$$dr_t = \\kappa(\\theta - r_t)\\,dt + \\sigma\\sqrt{r_t}\\,dW_t$$

where:
- $\\kappa > 0$: mean-reversion speed
- $\\theta > 0$: long-run equilibrium rate
- $\\sigma > 0$: volatility of the short rate
- $W_t$: standard Brownian motion

**The Feller Condition:** $2\\kappa\\theta \\geq \\sigma^2$ ensures $r_t > 0$ almost surely. When violated, the process can touch zero.

### Affine Bond Pricing

The CIR model admits a closed-form zero-coupon bond price $P(t,T) = A(\\tau)\\,e^{-B(\\tau)\\,r_t}$ where $\\tau = T - t$ and $\\gamma = \\sqrt{\\kappa^2 + 2\\sigma^2}$:

$$B(\\tau) = \\frac{2(e^{\\gamma\\tau}-1)}{(\\gamma+\\kappa)(e^{\\gamma\\tau}-1)+2\\gamma}$$

$$\\log A(\\tau) = \\frac{2\\kappa\\theta}{\\sigma^2}\\log\\left[\\frac{2\\gamma\\,e^{(\\kappa+\\gamma)\\tau/2}}{(\\gamma+\\kappa)(e^{\\gamma\\tau}-1)+2\\gamma}\\right]$$

The continuously compounded yield is: $y(\\tau) = -\\frac{\\log P}{\\tau} = \\frac{B(\\tau)\\,r_t - \\log A(\\tau)}{\\tau}$

This affine structure -- yields linear in the state variable $r_t$ -- is what makes CIR analytically tractable.
""",
    'CIR Calibration': """---
## Section 3: Model Calibration

CIR parameters $(\\kappa, \\theta, \\sigma)$ are unobservable and must be inferred from data. Three methods of increasing sophistication:

### 3.1 OLS (Baseline)
Linearise the Euler discretisation: $\\Delta r_t \\approx \\kappa\\theta\\Delta t - \\kappa r_t \\Delta t + \\varepsilon_t$. Run OLS on $\\Delta r \\sim r$; recover parameters from coefficients. Fast but ignores the exact transition density.

### 3.2 Maximum Likelihood (Exact)
The CIR transition density is a **scaled non-central chi-squared**:
$$2c\\,r_{t+\\Delta t} \\mid r_t \\sim \\chi^2(\\nu,\\,\\lambda)$$
where $c = \\frac{2\\kappa}{\\sigma^2(1-e^{-\\kappa\\Delta t})}$, $\\nu = \\frac{4\\kappa\\theta}{\\sigma^2}$, $\\lambda = 2c\\,r_t\\,e^{-\\kappa\\Delta t}$.

Multi-start L-BFGS-B maximises $\\ell = \\sum_t \\log p(r_{t+1}|r_t;\\,\\kappa,\\theta,\\sigma)$.

### 3.3 Kalman Filter (Gold Standard)
Treats $r_t$ as a **latent state** observed through all maturities simultaneously:
- **State:** $r_{t+1} = e^{-\\kappa\\Delta t}r_t + (1-e^{-\\kappa\\Delta t})\\theta + \\eta_t$
- **Observation:** $y_i(t) = \\frac{B(\\tau_i)}{\\tau_i}r_t - \\frac{\\log A(\\tau_i)}{\\tau_i} + \\varepsilon_{i,t}$

This uses all 9 maturities as simultaneous signals, models measurement noise explicitly, and the RTS smoother gives the full posterior over $r_t$.
""",
    'Yield Curve Prediction': """---
## Section 4: Base CIR Yield Curve Prediction

**Strategy:** From the observed 3-month yield, invert the CIR bond formula to recover the implied short rate $r_t$, then predict all 8 other maturities using the closed-form yield curve.

**Inversion:** Given $y_{3M}$ and the calibrated $B(0.25)$, $\\log A(0.25)$:
$$r_t = \\frac{y_{3M} \\cdot 0.25 + \\log A(0.25)}{B(0.25)}$$

**Prediction:** $\\hat{y}(\\tau) = \\frac{B(\\tau)\\,r_t - \\log A(\\tau)}{\\tau}$ for $\\tau \\in \\{0.5, 0.75, 1, 2, 5, 10, 20, 30\\}$ years.

This is a pure model-driven approach -- no regression, no curve fitting. The entire yield curve is determined by one number ($r_t$) and three parameters.
""",
    'CIR++ Infrastructure': """---
## Section 5: CIR++ Infrastructure -- Forward Rates & Shift Function

### Why CIR++ Is Needed
Base CIR cannot reproduce today's observed yield curve exactly -- there is always a fitting error at $t=0$. For pricing and hedging, this is unacceptable.

### The Brigo-Mercurio (2001) Extension
CIR++ adds a deterministic shift $\\varphi(t)$ to the base CIR process:
$$r_t^{++} = x_t + \\varphi(t)$$
where $x_t$ follows the standard CIR dynamics and
$$\\varphi(t) = f^M(0,t) - f^{CIR}(0,t;\\,x_0)$$

Here $f^M(0,t) = -\\frac{\\partial}{\\partial T}\\log P^M(0,T)\\big|_{T=t}$ is the market instantaneous forward rate (obtained from Nelson-Siegel interpolation of observed yields), and $f^{CIR}(0,t)$ is the model-implied forward rate.

**Key property:** $\\varphi(t)$ is chosen so that $P^{++}(0,T) = P^M(0,T)$ for all $T$ -- the model reprices every market bond exactly at $t=0$.
""",
    'CIR++ Bond Pricing': """---
## Section 6: CIR++ Bond Pricing & Prediction

The CIR++ bond price uses the ratio formula:
$$P^{++}(t,T) = \\frac{P^M(0,T)}{P^M(0,t)} \\cdot \\frac{P^{CIR}(t,T;\\,x_t)}{P^{CIR}(0,T;\\,x_0)}$$

where:
- $P^M(0,\\cdot)$ = market discount factors (from Nelson-Siegel bootstrapping)
- $P^{CIR}(\\cdot)$ = base CIR bond prices
- $x_t = r_t - \\varphi(t)$ = the CIR latent process

This preserves the initial curve fit while allowing dynamic evolution driven by the CIR diffusion. The shift function absorbs all model misfit at $t=0$.
""",
    'CIR Jump-Diffusion': """---
## Section 7: CIR Jump-Diffusion -- SDE & Simulation

### Why Jumps Are Needed
The CIR diffusion $dr = \\kappa(\\theta-r)dt + \\sigma\\sqrt{r}\\,dW$ is **continuous** by construction. But real short rates exhibit sudden discontinuous jumps from Fed rate decisions, crisis events, and policy surprises. Base CIR systematically underestimates tail risk.

### The CIR-J SDE (Duffie-Pan-Singleton, 2000)
$$dr_t = \\kappa(\\theta - r_t)\\,dt + \\sigma\\sqrt{r_t}\\,dW_t + dZ_t$$

where $Z_t = \\sum_{i=1}^{N_t} J_i$ is a compound Poisson process:
- $N_t \\sim \\text{Poisson}(\\lambda)$: jump arrival intensity
- With probability $p_{up}$: $J \\sim +\\text{Exp}(\\mu_{up})$ (up-jump)
- With probability $1-p_{up}$: $J \\sim -\\text{Exp}(\\mu_{down})$ (down-jump)

**Detection:** Jumps are identified from historical data using dual-method consensus (z-score + quantile) to reduce false positives. Parameters $\\lambda$, $\\mu_{up}$, $\\mu_{down}$, $p_{up}$ are estimated from detected jumps.

**Simulation:** Euler-Maruyama with Poisson jump overlay, reflecting barrier at $r=0$.
""",
    'CIR-J Bond Pricing (Ricatti ODEs)': """---
## Section 8: CIR-J Affine Bond Pricing -- Ricatti ODEs

### Duffie-Pan-Singleton (2000) Bond Pricing
Under CIR-J, bond prices retain the exponential-affine form:
$$P(t,T) = e^{\\alpha(\\tau) + \\beta(\\tau)\\,r_t}$$

The functions $\\alpha(\\tau)$, $\\beta(\\tau)$ satisfy coupled Ricatti ODEs:

$$\\frac{d\\beta}{d\\tau} = -1 - \\kappa\\beta + \\frac{\\sigma^2}{2}\\beta^2$$

$$\\frac{d\\alpha}{d\\tau} = \\kappa\\theta\\beta + \\lambda\\left[\\frac{p_{up}}{1-\\mu_{up}\\beta} + \\frac{1-p_{up}}{1+\\mu_{down}\\beta} - 1\\right]$$

with $\\beta(0)=0$, $\\alpha(0)=0$.

**Why affine structure is preserved:** The Laplace transform of $\\text{Exp}(\\mu)$ is $E[e^{\\beta J}] = 1/(1-\\mu\\beta)$, which is rational in $\\beta$. This keeps the ODE system tractable.

**Domain constraint:** $1 + \\mu_{down}\\beta > 0$ limits the maximum maturity. Solved using scipy's stiff-capable Radau solver with domain event termination.

**Critical validation:** When $\\lambda=0$, the ODE reduces to base CIR closed-form (verified to < 0.01 bps).
""",
    'CIR-J Prediction': """---
## Section 9: CIR-J Prediction & Stress Period Analysis

Using the ODE-based bond pricer, we invert the CIR-J yield formula at the 3M maturity to recover $r_t$, then predict the full curve -- analogous to the base CIR approach but with jump-augmented pricing.

**Regime analysis:** Test days are split into JUMP days (flagged by the detector) and CALM days. We compare Base CIR vs CIR-J performance separately on each regime to quantify the value of the jump component.
""",
    'Grand Model Comparison': """---
## Section 10: Grand Model Comparison

All three models (Base CIR, CIR++, CIR-J) compared head-to-head across:
- Every maturity (6M through 30Y)
- R², RMSE, MAE, bias, max error, hit rate (<10bp)
- Pairwise Diebold-Mariano tests for statistical significance
- Model scorecard with complexity/performance trade-off
""",
    'Parameter Stability': """---
## Section 11: Rolling Parameter Stability

A key assumption of the CIR model is **time-homogeneous parameters**. We test this by calibrating $(\\kappa, \\theta, \\sigma)$ on rolling 1-year windows (monthly step), tracking:
- Parameter trajectories and confidence bands
- Feller condition violations over time
- Parameter cross-correlations
- Coefficient of variation as a stability metric

High parameter instability undermines the constant-parameter assumption and motivates regime-switching extensions.
""",
    'Critical Analysis': """---
## Section 12: Mathematical Limitations

Rigorous quantitative analysis of four fundamental CIR limitations:

1. **Single-factor constraint:** PCA on the yield matrix reveals how much variance is missed by using only one factor (level). PC2 (slope) and PC3 (curvature) are systematically unmodelled.

2. **Constant parameter assumption:** Rolling calibration reveals parameter drift. The cost is quantified by comparing rolling vs full-sample parameter deviations.

3. **Gaussian discretisation error:** Euler-Maruyama replaces the exact non-central chi-squared transition with a Gaussian approximation. Wasserstein distance measured at multiple horizons.

4. **Zero lower bound:** CIR is non-negative by construction but cannot produce negative rates. Behaviour analysed in hypothetical low-rate scenarios.
""",
    'Practical Limitations': """---
## Section 13: Practical Limitations & Final Report

Market-structure limitations and production considerations:
- **Curve shape dependence:** Performance breakdown by yield curve shape (normal, flat, inverted, humped)
- **Liquidity premium bias:** Systematic spread between model-implied and actual yields
- **Overfitting risk:** Train/validation/test R² gap analysis
- **Input sensitivity:** Monte Carlo noise amplification from 3M input to predicted yields
- **Final synthesis report** with model recommendations by use case
""",
    'MAIN EXECUTION BLOCK': """---
## Main Execution

Full pipeline: data loading → EDA → calibration (OLS/MLE/Kalman) → base CIR prediction → CIR++ extension → jump detection → CIR-J pricing → grand comparison → rolling stability → limitation analysis → final report.
""",
}


COLAB_SETUP = '''import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["numpy","pandas","scipy","matplotlib","seaborn",
            "statsmodels","filterpy","scikit-learn","tqdm","tabulate"]:
    install(pkg)
print("All packages installed.")

try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    pass

import os
os.makedirs("data", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)
print("Directories ready. Place train.csv and test.csv in data/.")
'''


def read_main_py(path='main.py'):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_sections(source: str):
    """Split main.py into logical code chunks by top-level '# ---' section markers."""
    lines = source.split('\n')
    chunks = []
    current_chunk = []
    current_label = None

    for line in lines:
        stripped = line.strip()
        # Only split on UNINDENTED markers (column 0) to avoid breaking methods
        is_top_level = (not line[:1].isspace()) and line.startswith('# ---')
        if is_top_level and stripped.endswith('---') and len(stripped) > 10:
            if current_chunk:
                chunks.append((current_label, '\n'.join(current_chunk)))
                current_chunk = []
            current_label = stripped[5:-3].strip()
        current_chunk.append(line)

    if current_chunk:
        chunks.append((current_label, '\n'.join(current_chunk)))

    return chunks


def build_notebook(main_path='main.py', out_path='main.ipynb'):
    print(f"Building notebook from {main_path}...")
    source = read_main_py(main_path)
    chunks = split_into_sections(source)

    cells = []

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + '\n' for ln in COLAB_SETUP.strip().split('\n')],
    })

    for label, code in chunks:
        if label and label in MARKDOWN_INSERTS:
            md_text = MARKDOWN_INSERTS[label].strip()
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [ln + '\n' for ln in md_text.split('\n')],
            })

        code_lines = code.strip()
        if code_lines:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [ln + '\n' for ln in code_lines.split('\n')],
            })

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "colab": {
                "provenance": [],
                "toc_visible": True,
            },
        },
        "cells": cells,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Notebook saved: {out_path} ({len(cells)} cells)")
    print(f"  - {sum(1 for c in cells if c['cell_type']=='markdown')} markdown cells")
    print(f"  - {sum(1 for c in cells if c['cell_type']=='code')} code cells")
    return out_path


if __name__ == "__main__":
    build_notebook()
