import numpy as np
import math
from scipy.stats import norm
import pdb

import argparse
''' chance of missing signal variables at the first sampling '''
def get_saturation_prob(n, R, k, alpha):
    p = 0.5 + 0.5 * ((R - alpha) / R) ** (n - 1)
    prob_1 = p ** k
    prob_2 = k * (1 - p) * p ** (k - 1)
    return 1 - prob_1 - prob_2

def miss_prop_at_t(x, n, R, k, sigma, signal, alpha, T, t):
    print(">", x, n, R, k, sigma, signal, alpha, T, t)
    order_expect = -norm.ppf(((k - 1) / 2 - 0.375) / (k + 1 - 2 * 0.375))
    sigma_x = sigma * np.sqrt(1 / t + math.pi * (n - 1) * (1 - alpha) / (2 * k * t * (R - alpha)))
    percentile_1 = -x / sigma_x
    percentile_2 = -(x / sigma_x - order_expect)
    p = 0.5 + 0.5 * ((R - alpha) / R) ** (n - 1)
    prob_1 = p ** k
    prob_2 = k * (1 - p) * p ** (k - 1)
    print("Saturation Probability", (1 - prob_1 - prob_2))


    return prob_1 * norm.cdf(percentile_1) + prob_2 * norm.cdf(percentile_2) + (1 - prob_1 - prob_2)


def miss_prop_after_t(theta, Tau, n, R, k, sigma, signal, alpha, T, t):
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


def find_best_exploration_period(signal, alpha, init_threshold, num_features, cs_params, total_samples, target_miss_probability, sigma=None):
    '''
        signal : signal level (for correlation recommended is 0.2)
        alpha : proportion of signal variables . More of an upper bound . give reasonable value. (eg. 0.5% = 0.005)
        init_threshold : also called tau, threshold immediately after the exploration period.
             >> should be low enough so that we can have a small exploration period

        num_features : number of variables to insert in count sketch
        cs_params : dictionary where you can store the params of cs , expected keys K, R
        total_samples: total number of samples that will be added
        target_miss_probability : target miss probability (this should be low enough)
        
    '''
    #miss_prop_at_t(x, n=n, R=R, k=k, sigma=sigma, signal=signal, alpha=alpha, T=T, t=t):
    print(signal, alpha, init_threshold, num_features, cs_params, total_samples, target_miss_probability)
    if sigma is None: # This is CORRELATION CALL
        sigma = np.sqrt(1 + signal**2) # TODO This is true only for correlations
    start = 1
    end = total_samples
    while(end > start + 1):
      mid = int((start + end)/2)
      x = signal-init_threshold * total_samples / mid #TODO (zhenwei verify)
      prob = miss_prop_at_t(x, n=num_features, R=cs_params["R"],
                            k=cs_params["K"], sigma=sigma, signal=signal,
                            alpha=alpha, T=total_samples, t=mid)

      print(start, end, mid, prob)
      if prob > target_miss_probability:
        start = mid
      else:
        end = mid 
    return start

def find_best_exploration_period_m2(signal, alpha, init_threshold, num_features, cs_params, total_samples, target_miss_probability, sigma=None):
    '''
        signal : signal level (for correlation recommended is 0.2)
        alpha : proportion of signal variables . More of an upper bound . give reasonable value. (eg. 0.5% = 0.005)
        init_threshold : also called tau, threshold immediately after the exploration period.
             >> should be low enough so that we can have a small exploration period

        num_features : number of variables to insert in count sketch
        cs_params : dictionary where you can store the params of cs , expected keys K, R
        total_samples: total number of samples that will be added
        target_miss_probability : target miss probability (this should be low enough)
        
    '''
    #miss_prop_at_t(x, n=n, R=R, k=k, sigma=sigma, signal=signal, alpha=alpha, T=T, t=t):
    print(signal, alpha, init_threshold, num_features, cs_params, total_samples, target_miss_probability)
    if sigma is None:
        sigma = np.sqrt(1 + signal**2) # TODO This is true only for correlations
    start = 1
    end = total_samples
    values = []
    mid = start
    while(end > start + 1):
      mid = int((start + end)/2)
      x = signal-init_threshold * total_samples / mid #TODO (zhenwei verify)
      prob = miss_prop_at_t(x, n=num_features, R=cs_params["R"],
                            k=cs_params["K"], sigma=sigma, signal=signal,
                            alpha=alpha, T=total_samples, t=mid)

      values.append((mid, prob))
      if prob > target_miss_probability:
        start = mid
      else:
        end = mid 
    print(values)
    if total_samples - end < 2:
      print("The probability limit Failed! finding the approximate spurt point using error : 0.025")
      values = np.sort(values)
      selected = values[values[:,0] <= (values[:,0][-1] + 0.025)][0]
      print("selected", selected)
      return int(selected[1])
    else:
      return mid

def find_best_theta(signal, alpha, init_threshold, num_features, cs_params, total_samples, exploration_samples, target_miss_probability, sigma=None):
  #def miss_prop_after_t(theta, Tau, n, R, k, sigma, signal, alpha, T, t):
    start = 0.01 * signal
    end = 0.99 * signal # theta has to be less than signal
    # least count 
    least_count = 0.001 * signal # 


    while(end > start + least_count):
      mid = (start + end)/2
      if sigma is None: #correlations
          sigma = np.sqrt(1 + signal**2) # TODO This is true only for correlations
      prob = miss_prop_after_t(theta = mid, Tau = init_threshold,
                              n=num_features, R=cs_params["R"],
                              k=cs_params["K"], sigma=sigma,
                              signal=signal, alpha=alpha, T=total_samples,
                              t=exploration_samples)
      print(start, end, mid, prob)
      if prob < target_miss_probability:
        start = mid
      else:
        end = mid 
    return start


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--signal', action="store", dest="signal", type=float, required=True,  help="")
    parser.add_argument('--alpha', action="store", dest="alpha", type=float, required=True,  help="")
    parser.add_argument('--init_thold', action="store", dest="init_thold", type=float, required=True,  help="")
    parser.add_argument('--num_samples', action="store", dest="num_samples", type=float, required=True,  help="")
    parser.add_argument('--num_entries', action="store", dest="num_entries", type=float, required=True,  help="")
    parser.add_argument('--cs_r', action="store", dest="cs_r", type=int, required=True,  help="")
    parser.add_argument('--cs_k', action="store", dest="cs_k", type=int, required=True,  help="")
    parser.add_argument('--delta', action="store", dest="delta", type=float, required=True,  help="")
    parser.add_argument('--delta_star', action="store", dest="delta_star", type=float, required=True,  help="")
    res = parser.parse_args()
    signal = res.signal
    alpha = res.alpha
    delta_star = res.delta_star
    delta = res.delta
    cs_k = res.cs_k
    cs_r = res.cs_r
    num_entries = res.num_entries
    num_samples = res.num_samples
    init_thold = res.init_thold

    exp = find_best_exploration_period_m2(signal, alpha, init_thold, num_entries, {"K":cs_k, "R":cs_r }, num_samples, delta)
    #exp = 0.2*num_samples
    print("----------------- Exploration Period:", exp)
    theta  = find_best_theta(signal, alpha, init_thold, num_entries, {"K":cs_k, "R":cs_r }, num_samples, exp, delta_star)
    print("----------------- theta:", theta)
    #for t in range(1, T, 500):
    #  print('Probability of missing a signal variable at exploration ',t,
    #        miss_prop_at_t(signal-Tau*T/t, n, R, k, sigma, signal, alpha, T, t))
    
    
    
    '''
    Find t, such that miss_prop_at_t around 0.05
    Find theta, such that miss_prop_after_t around 0.15
    Sum up, it is less than 0.2
    '''
    




