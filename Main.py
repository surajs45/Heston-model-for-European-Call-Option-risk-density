import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def heston_pdf(S, K, t, r, q, v0, kappa, theta, sigma, rho):
    
    # Compute the characteristic function of Heston model
    def char_func(u):
        omega = kappa - sigma * 1j * rho * u
        alpha = 1 - omega**2 / (4 * theta**2)
        beta = (kappa * theta - sigma**2 * rho * u * 1j) / (2 * theta**2)
        gamma = sigma**2 / (2 * theta**2)
        D = np.sqrt(omega**2 - 4 * theta**2 * (alpha - 1))
        G = (omega - D) / (omega + D)
        C = (1 - 1j * u * beta - gamma * u**2) / (1 - G * np.exp(-D * t))
        return np.exp (1j * u * (np.log(S) + (r - q)* t)) * C
    
    def pdf(x):
        u = np.log(K / x)
        return (1 / (x * K * np.pi)) * np.imag(char_func(u - 1j) / (u - 1j))
    
    return pdf

# Sample values for testing
S = 100 # spot price
K = 110 # strike price
t = 1 # time to expiration
r = 0.05 # risk-free rate
q = 0.01 # dividend yield
v0 = 0.05 # initial volatility
kappa = 1 # mean reversion rate
theta = 0.05 # mean reversion level
sigma = 0.3 # volatility of volatility
rho = -0.5 # correlation between spot price and volatility

pdf = heston_pdf(S, K, t, r, q, v0, kappa, theta, sigma, rho)

S_range = np.linspace(0.9 * S, 1.1 * S, 1000)

density = pdf(S_range)

plt.plot(S_range, density)
plt.xlabel("Spot Price")
plt.ylabel("Density")
plt.title('Risk Neutral Density Function under the Heston Model')
plt.show()
