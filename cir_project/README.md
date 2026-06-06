# Stochastic Interest Rate Modelling
Google collab link -> https://colab.research.google.com/drive/1Vbq_IVRIJ2qDAWZIakEn2haFPHLTOZBi?usp=sharing
### Models Implemented

| Model | Description | Parameters |
|-------|-------------|------------|
| **Base CIR** | Cox-Ingersoll-Ross (1985) | kappa, theta, sigma |
| **CIR++** | Brigo-Mercurio shift extension | kappa, theta, sigma + phi(t) |
| **CIR-J** | Jump-diffusion (Duffie-Pan-Singleton 2000) | kappa, theta, sigma, lambda, mu_up, mu_down |

### Calibration Methods

- **OLS** -- Ordinary Least Squares baseline
- **MLE** -- Maximum Likelihood with NCX2 transition density
- **Kalman Filter** -- State-space estimation with RTS smoother

### Project Structure

```
cir_project/
  main.py              # Full pipeline (~5900 lines)
  main.ipynb           # Notebook version (auto-generated)
  build_notebook.py    # main.py -> main.ipynb converter
  requirements.txt     # Python dependencies
  data/
    train.csv          # Training yield curves
    test.csv           # Test yield curves
  outputs/
    plots/             # All generated plots
    results/           # CSV tables, markdown reports
  tests/
    generate_synthetic_data.py
```

### Quick Start

```bash
pip install -r requirements.txt
python main.py
python build_notebook.py
```

### Notebook (Colab)

1. Run `python build_notebook.py` to generate `main.ipynb`
2. Upload to Google Colab
3. Upload `train.csv` and `test.csv` to `data/`
4. Run all cells

### Key Results

- Out-of-sample R^2 > 0.85 across 8 maturities (6M--30Y)
- CIR++ fits the initial term structure to < 2 bps
- CIR-J improves performance during jump/stress periods
- Rolling calibration reveals time-varying Feller condition violations
- PCA confirms single-factor captures ~95% of yield variance
