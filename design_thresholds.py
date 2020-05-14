import numpy as np
import math
from scipy.stats import norm


def mu_hat_approx(x, n=n, R=R, k=k, sigma=sigma, signal=signal, alpha=alpha, T=T, t=t):
    order_expect = -norm.ppf(((k - 1) / 2 - 0.375) / (k + 1 - 2 * 0.375))
    sigma_x = sigma * np.sqrt(1 / t + math.pi * (n - 1) * (1 - alpha) / (2 * k * t * (R - alpha)))
    percentile_1 = -x / sigma_x
    percentile_2 = -(x / sigma_x - order_expect)
    p = 0.5 + 0.5 * ((R - alpha) / R) ** (n - 1)
    prob_1 = p ** k
    prob_2 = k * (1 - p) * p ** (k - 1)

    return prob_1 * norm.cdf(percentile_1) + prob_2 * norm.cdf(percentile_2) + (1 - prob_1 - prob_2)


def miss_prop_approx(theta, Tau=Tau, n=n, R=R, k=k, sigma=sigma, signal=signal, alpha=alpha, T=T, t=t):
    order_expect = -norm.ppf((k / 2 - 0.375) / (k + 1 - 2 * 0.375))
    var = math.pi * (n - 1) * (1 - alpha) * sigma ** 2 / (2 * k * (R - alpha))

    nu_0 = t / T * signal
    nu_1 = (signal - theta) / T
    nu_0_tilde = t / T * (signal - order_expect * np.sqrt(var / t))
    nu_1_tilde = 1 / T * (signal - theta - order_expect * np.sqrt(var) * (np.sqrt(t + 1) - np.sqrt(t)))

    omega_0 = np.sqrt(t / T ** 2 * (1 + var))
    omega_1 = omega_0 / np.sqrt(t)
    omega_0_tilde = omega_0
    omega_1_tilde = omega_1

    a1 = np.exp(-nu_0 ** 2 / (2 * omega_0 ** 2) + 2 * Tau * nu_1 / omega_1 ** 2 + (
                nu_0 - 2 * nu_1 * omega_0 ** 2 / omega_1 ** 2) ** 2 / (2 * omega_0 ** 2)) * \
         norm.cdf(((nu_0 - 2 * nu_1 * omega_0 ** 2 / omega_1 ** 2) - Tau) / omega_0)

    a2 = np.exp(-nu_0 ** 2 / (2 * omega_0_tilde ** 2) + 2 * Tau * nu_1_tilde / omega_1_tilde ** 2 +
                (nu_0_tilde - 2 * nu_1_tilde * omega_0_tilde ** 2 / omega_1_tilde ** 2) ** 2 / (
                            2 * omega_0_tilde ** 2)) * \
         norm.cdf(((nu_0_tilde - 2 * nu_1_tilde * omega_0_tilde ** 2 / omega_1_tilde ** 2) - Tau) / omega_0_tilde)

    p = 0.5 + 0.5 * ((R - alpha) / R) ** (n - 1)
    prob_1 = p ** k
    prob_2 = k * (1 - p) * p ** (k - 1)

    return a1 * prob_1 + a2 * prob_2


if __name__ == '__main__':
    '''
    n: number of covariance entries
    R: length of each hash table
    k: number of hash tables
    alpha: proportion of signal variables (upper bound)
    signal: the signal level (signal variables should be larger than 0.2)
    sigma: variance of signal variable (sigma = (1 + signal)^2)
    t: length of exploration period
    T: total sample size
    '''

    n = NUM_FEATURES * (NUM_FEATURES - 1) / 2
    R = 12000
    k = 5
    alpha = 2.5 * 10 ** (-3)
    signal = 0.2
    sigma = 1.44
    t = 1250
    T = 5000

    '''
    Tau is the sampling threshold you choose right after the exploration period
    '''
    Tau = 0.03

    '''
    theta is the step size to raise the sampling thresholds.
    sampling thresholds at time t_0, Tau_{t_0} = Tau + theta*(t_0-t)/T 
    '''
    theta = 0.15

    print('Probability of missing a signal variable during sampling: ',
          mu_hat_approx(signal-Tau*T/t) + miss_prop_approx(theta))



