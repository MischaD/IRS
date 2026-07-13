import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import fsolve

# ==========================================
# 1. Math & Stirling Approximation Helpers
# ==========================================

class LogFactorial:
    def __init__(self, max_value):
        """Initialize the LogFactorial class with a precomputed table up to max_value."""
        self.max_value = max_value
        self.logfactorial_lookup = self.compute_logfactorial_lookup(max_value)

    def compute_logfactorial_lookup(self, max_value):
        logfactorial_lookup = [0] * (max_value + 1)
        for i in range(2, max_value + 1):
            logfactorial_lookup[i] = logfactorial_lookup[i - 1] + np.log(i)
        return logfactorial_lookup

    def extend_lookup(self, new_max_value):
        if new_max_value > self.max_value:
            print(f"[INFO] Extending LogFactorial lookup table to {new_max_value}")
            for i in range(self.max_value + 1, new_max_value + 1):
                self.logfactorial_lookup.append(self.logfactorial_lookup[i - 1] + np.log(i))
            self.max_value = new_max_value

    def __call__(self, x):
        if x > self.max_value:
            self.extend_lookup(x)
        return self.logfactorial_lookup[x]


# Global instance
logfactorial = LogFactorial(int(5e4))


def log_binom(n, k): 
    return logfactorial(n) - logfactorial(k) - logfactorial(n - k)


def log_stirling_second_kind_approx(n, k):
    if n == k: 
        return 0 
    assert k > 0 and k < n 

    v = n / k

    def G_func(G):
        return G - v * np.exp(G - v)

    G_initial_guess = 0.5
    G = fsolve(G_func, G_initial_guess)[0]

    part1 = 0.5 * np.log((v - 1) / (v * (1 - G)))
    part2 = (n - k) * np.log(((v - 1) / (v - G)))
    part3 = n * np.log(k) - k * np.log(n) + k * (1 - G)

    approximation = part1 + part2 + part3 + log_binom(n, k)
    return approximation


def log_compute_formula(s, k, n): 
    logstir = log_stirling_second_kind_approx(n, k)
    return logstir + logfactorial(s) - logfactorial(s - k) - n * np.log(s)


# ==========================================
# 2. Main IRS Metric Class
# ==========================================

class IRSMetric:
    def __init__(
        self, 
        alpha_e=0.05, 
        prob_tolerance=1e-10, 
        naive=False, 
        batch_size=512, 
        verbose=True
    ):
        self.alpha_e = alpha_e
        self.confidence = True
        self.prob_tolerance = prob_tolerance
        self.naive = naive
        self.batch_size = batch_size
        self.verbose = verbose

    def compute_irs_inf(self, n_train_max, n_sampled, k_measured): 
        n_train = n_train_max
        alpha_e = self.alpha_e
        confidence = self.confidence 
        prob_tolerance = self.prob_tolerance
        naive = self.naive
        
        alpha_of_IRS_alpha = n_sampled / n_train
        if self.verbose:
            print(f"Maximum Possible Number of Train Images: {n_train_max}\nSampled images: {n_sampled}\nLearned images: {k_measured}")
            print(f"IRS (alpha={alpha_of_IRS_alpha:.2f}): {k_measured / n_train_max}")
        
        if naive: 
            probs = []
            n_train_ests = [*range(k_measured, n_train_max)]
            for n_train_est in n_train_ests: 
                alpha_of_IRS_alpha = n_sampled / n_train_est 
                irs_alpha = np.exp(log_compute_formula(s=n_train_est, k=k_measured, n=n_sampled))
                probs.append(irs_alpha)
                if len(probs) > 2 and probs[-2] > probs[-1] and irs_alpha < prob_tolerance: 
                    break
            probs = np.array(probs)

            irs_inf = np.argmax(probs)
            n_learned_pred = n_train_ests[irs_inf] 

        else: 
            low = k_measured
            high = n_train_max
            while low <= high:
                mid = (low + high) // 2

                if high - low == 2: 
                    break

                prob_mid_m1 = log_compute_formula(s=mid - 1, k=k_measured, n=n_sampled)
                prob_mid = log_compute_formula(s=mid, k=k_measured, n=n_sampled)
                prob_mid_p1 = log_compute_formula(s=mid + 1, k=k_measured, n=n_sampled)

                if prob_mid > max(prob_mid_m1, prob_mid_p1): 
                    break  
                if prob_mid >= prob_mid_p1: 
                    high = mid
                else: 
                    low = mid

            prob_mid_l1 = log_compute_formula(s=mid - 1, k=k_measured, n=n_sampled)
            prob_mid = log_compute_formula(s=mid, k=k_measured, n=n_sampled) 
            prob_mid_u1 = log_compute_formula(s=mid + 1, k=k_measured, n=n_sampled)
            n_learned_pred = mid - 1 + np.argmax(
