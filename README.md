# Swaption Implied Volatility under Negative Rates


https://github.com/user-attachments/assets/79393005-4a59-4f58-85b8-c4ecacaeea80


This repository implements a Python framework for modeling negative interest rates and computing swaption and caplet/floorlet implied volatilities using the Hull–White one-factor model and Black-76 methodology.

## Overview

- **Model:** Hull–White one-factor short-rate model, calibrated to fit the initial zero-coupon curve.
- **Features:**
  - Monte Carlo simulation of short-rate paths (Euler–Maruyama).
  - Analytical zero-coupon bond pricing.
  - Caplet and floorlet valuation as bond options under the $T_2$-forward measure.
  - Recovery of Black‑76 implied volatilities via root-finding.

## Theory & Mathematics

### 1. Hull–White Model Dynamics

```math
\mathrm{d}r(t) = \lambda\bigl(\theta(t) - r(t)\bigr)\mathrm{d}t + \eta\,\mathrm{d}W(t)
```
- $\lambda$: mean-reversion speed
- $\theta(t)$: time-dependent drift matching $P(0,T)$
- $\eta$: volatility

### 2. Calibration to Initial Curve

Instantaneous forward rate:
```math
f(0,T) = -\frac{\partial}{\partial T} \ln P(0,T)
```
Numerically via central difference with $\delta=10^{-4}$.

Drift term:
```math
\theta(t) = \frac{1}{\lambda} \frac{\partial f(0,t)}{\partial t} + f(0,t) + \frac{\eta^2}{2\lambda^2}(1 - e^{-2\lambda t}).
```

### 3. Zero-Coupon Bond Pricing

Affine form under Hull–White:
```math
P(T_1,T_2) = \exp\bigl(A(T_1,T_2) + B(T_1,T_2)\,r(T_1)\bigr)
```
where
```math
B(T_1,T_2) = \frac{1}{\lambda}(e^{-\lambda\tau} - 1), \quad \tau = T_2 - T_1
```
and
```math
A(T_1,T_2) = \int_{0}^{\tau} \lambda\,\theta(T_2 - u)B(u)\,du + \frac{\eta^2}{4\lambda^3}(e^{-2\lambda\tau}(4e^{\lambda\tau} -1)-3) + \frac{\eta^2\tau}{2\lambda^2}.
```

### 4. Caplet/Floorlet Pricing

Forward LIBOR rate:
```math
L(T_1,T_2) = \frac{1}{T_2 - T_1}\Bigl(\frac{1}{P(T_1,T_2)} - 1\Bigr)
```
Caplet payoff at $T_2$:
```math
N(T_2 - T_1)\max(L-K,0) = N^*\max(P(T_1,T_2) - K^*,0)
```
with
```math
K^* = 1 + K(T_2 - T_1), \quad N^* = N K^*.
```
Under the $T_2$‑forward measure:
```math
C_{\rm caplet} = N^* P(0,T_2) \mathbb{E}^{T_2}[\max(X - 1/K^*,0)], \quad X=P(T_1,T_2).
```
Closed-form using normal CDFs:
```math
C_{\rm caplet} = N^*P(0,T_2)[\Phi(-d_2) - K^*{}^{-1}e^{A+B\mu}\Phi(-d_1)]
```
with
```math
d_1 = \frac{\ln(K^*e^{-A}) - B\mu}{v} + \frac{v}{2}, \quad d_2 = d_1 - v.
```

### 5. Implied Volatility: Black‑76

Solve for $\sigma$ in
```math
BS(\sigma) - \text{ModelPrice} = 0,
```
where
```math
BS_{\rm call} = P(0,T_2)[\Phi(d_1)F - \Phi(d_2)K], \quad d_{1,2} = \frac{\ln(F/K) \pm 0.5\sigma^2\tau}{\sigma\sqrt{\tau}}.
```

## Installation

```bash
git clone https://github.com/shubh123a3/Swapstion_Implied-Volality_negative-rates.git
cd Swapstion_Implied-Volality_negative-rates
pip install -r requirements.txt
```

## Usage

Edit parameters in `main()` and run:

```bash
python your_script.py
```
Plots and numerical outputs will be displayed for bond prices and caplet/floorlet implied vols.

## File Structure

- `hw_model.py`: Hull–White model and bond pricing
- `cap_floor.py`: Caplet/floorlet payoffs and implied vol routines
- `main.py`: Example scripts and plots

## License

MIT © Shubh

