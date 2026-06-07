## Practical Limitations

### 1. Curve Shape Limitations

The CIR model's single-factor structure constrains predicted yield curves
to a monotonic shape. During inverted-curve periods, all three models
show degraded performance. The model cannot organically produce inversion;
any apparent fit comes from the short-rate inversion proxy propagating
through the B(tau) function.

### 2. Liquidity Premium Bias

The model-implied spread shows a mean bias of
2.8 bps with std 6.9 bps.
This is systematic and represents an unmodeled liquidity/credit premium.
The autocorrelation of 0.981 suggests
persistent bias.

### 3. Overfitting Risk

The train-test R2 gap is -1.2410,
suggesting potential overfitting.
The CIR model's 3-parameter structure provides strong regularization.

### 4. Input Sensitivity

A 1.5 bps noise on the 3M input propagates to predicted yields with
amplification factors ranging from 0.23x
to 0.98x.
The most sensitive maturity is 6M.

### Production Recommendations

- Use **Base CIR** for quick estimates in normal curve environments
- Use **CIR++** when initial curve fit matters (pricing, hedging)
- Use **CIR-J** for stress testing and risk management
- Consider multi-factor models (AFNS, HJM) for production deployment
