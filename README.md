## Simulation of Spot Energy Prices with Seasonality, an Ornstein–Uhlenbeck Process, and Jump Diffusion

This project models the Italian electricity spot price (PUN) using a mean-reverting jump-diffusion model with a deterministic seasonal component. Although the model is calibrated to the Italian electricity market, the methodology can be generalized to other energy spot markets.

The main objective is to replicate and extend the methodology presented in Simulating Electricity Prices with Mean-Reversion and Jump-Diffusion.

## Dataset
PUN spot prices: daily Italian electricity spot prices from 01/01/2020 to 13/04/2026, downloaded from the Gestore dei Mercati Energetici (GME).
Futures prices: daily prices of monthly Italian electricity futures from 01/07/2025 to 13/04/2026, downloaded from the European Energy Exchange (EEX).
## Methodology
1. Fit a deterministic function based on sine and cosine terms to model the seasonal component.
2. Compute the deseasonalized log-price and fit an Ornstein–Uhlenbeck process to the residual component.
3. Apply Euler discretization to the continuous-time process.
4. Estimate the model parameters using Maximum Likelihood Estimation (MLE).
5. Perform Monte Carlo simulations under the real-world probability measure.
6. Compute the forward curve for futures contracts for a selected valuation date.
7. Estimate the market price of risk from observed futures prices by solving a linear system.
8. Apply Girsanov's theorem to obtain the risk-neutral parametrization.
9. Perform a second Monte Carlo simulation under the risk-neutral measure and compute the expected futures price at maturity.
10. Compare the theoretical futures prices with observed market prices.

## Results

The model is able to consistently reproduce the observed market futures prices, as shown in the figure below.
![Market Futures vs Theoretical Futures](images/results.png)
