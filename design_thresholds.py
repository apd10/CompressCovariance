import numpy as np
import math
from scipy.stats import norm

def mu_hat_approx(x, n, R, k):
    order_expect = norm.ppf((k/2-0.375)/(k+1-2*0.375))
    sigma_x = sigma*np.sqrt(1/t+math.pi*(n-1)*(1-alpha)/(2*k*t*(R-alpha)))
    percentile_1 = -x/sigma_x
    percentile_2 = -(x/sigma_x - order_expect)
    p = 0.5+0.5*((R-alpha)/R)**(n-1)
    prob_1 = p**k
    prob_2 = k*(1-p)*p**(k-1)
    return prob_1*norm.cdf(percentile_1) + prob_2*norm.cdf(percentile_2) + (1-prob_1-prob_2) 


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

    n = NUM_FEATURES * (NUM_FEATURES-1) / 2
    R = 12000
    k = 5
    alpha = 2.5*10**(-3)
    signal = 0.2
    sigma = 1.44
    t = 1250
    T = 5000

    '''
    Tau is the sampling threshold you choose
    '''
    Tau = 0.03

    print('Probability of missing a signal variable right after the exploration period: ',
          mu_hat_approx(signal-Tau*T/t, n, R, k))

