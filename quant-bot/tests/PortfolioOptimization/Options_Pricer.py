from FMNM.BS_pricer import BS_pricer
from FMNM.Parameters import Option_param
from FMNM.Processes import Diffusion_process

import numpy as np
import scipy.stats as ss
from scipy.integrate import quad
from functools import partial

import matplotlib.pyplot as plt

#%matplotlib inline
def pricer(S0=100, K=100, T=1, r=0.1, sig=0.2):
  # Creates the object with the parameters of the option
  opt_param = Option_param(S0, K, T, exercise="European", payoff="call")
  # Creates the object with the parameters of the process
  diff_param = Diffusion_process(r, sig)
  # Creates the pricer object
  BS = BS_pricer(opt_param, diff_param)
  BS.closed_formula()
  13.269676584660893
  BS.Fourier_inversion()
  13.269676584660623
  BS.MC(N=30000000, Err=True, Time=True)
  # output is: price, standard error and execution time
  # (array([13.26753511]), array([0.00294085]), 0.679786205291748)
  BS.mesh_plt()  # PDE method 

def bs_pricer(S0=100, K=100, T=1, r=0.1, sig=0.2):

  # S0 = 100.0  # spot stock price
  # K = 100.0  # strike
  # T = 1.0  # maturity
  # r = 0.1  # risk free rate
  # sig = 0.2  # diffusion coefficient or volatility
  
  call = BS_pricer.BlackScholes("call", S0, K, T, r, sig)
  put = BS_pricer.BlackScholes("put", S0, K, T, r, sig)
  print("Call price: ", call)
  print("Put price: ", put)
  
  # Put-Call Parity: should be equal
  print(call)
  print(put + S0 - K * np.exp(-r * T)
  
def monte_carlo_pricer(S0=100, K=100, T=1, r=0.1, sig=0.2):
  np.random.seed(seed=44)  # seed for random number generation
  N = 10000000  # Number of random variables
  
  W = ss.norm.rvs((r - 0.5 * sig**2) * T, np.sqrt(T) * sig, N)
  S_T = S0 * np.exp(W)
  
  call = np.mean(np.exp(-r * T) * np.maximum(S_T - K, 0))
  put = np.mean(np.exp(-r * T) * np.maximum(K - S_T, 0))
  call_err = ss.sem(np.exp(-r * T) * np.maximum(S_T - K, 0))  # standard error
  put_err = ss.sem(np.exp(-r * T) * np.maximum(K - S_T, 0))  # standard error

def binomial_tree_pricer(S0=100, K=100, T=1, r=0.1, sig=0.2):
  N = 15000  # number of periods or number of time steps
  payoff = "call"  # payoff
  
  dT = float(T) / N  # Delta t
  u = np.exp(sig * np.sqrt(dT))  # up factor
  d = 1.0 / u  # down factor
  
  V = np.zeros(N + 1)  # initialize the price vector
  
  # price S_T at time T
  S_T = np.array([(S0 * u**k * d ** (N - k)) for k in range(N + 1)])
  
  a = np.exp(r * dT)  # risk free compounded return
  p = (a - d) / (u - d)  # risk neutral up probability
  q = 1.0 - p  # risk neutral down probability
  
  if payoff == "call":
      V[:] = np.maximum(S_T - K, 0.0)
  else:
      V[:] = np.maximum(K - S_T, 0.0)
  
  for i in range(N - 1, -1, -1):
      # the price vector is overwritten at each step
      V[:-1] = np.exp(-r * dT) * (p * V[1:] + q * V[:-1])
  
  print("BS Tree Price: ", V[0])
  # BS Tree Price:  13.269537371978052
  
  %%timeit -n 20 -r 10
  S_T = np.array([(S0 * u**j * d ** (N - j)) for j in range(N + 1)])  # price S_T at time T
  # 2.64 ms ± 37.9 µs per loop (mean ± std. dev. of 10 runs, 20 loops each)
  %%timeit -n 20 -r 10
  S_0N = S0 / u**N
  S_T = np.array([S_0N * u ** (2 * j) for j in range(N + 1)])  # price S_T at time T
  # 1.59 ms ± 24.1 µs per loop (mean ± std. dev. of 10 runs, 20 loops each)
  %%timeit -n 20 -r 10
  S_0N = S0 / u**N
  S_T = np.array([S_0N * u ** (j) for j in range(0, 2 * N + 1, 2)])  # price S_T at time T
  # 1.49 ms ± 25 µs per loop (mean ± std. dev. of 10 runs, 20 loops each)
  # The last approach for the computation of is the fastest.
  
  
def pricer_model_limits(S0=100, K=100, T=1, r=0.1, sig=0.2):
  BS_sigma = partial(BS_pricer.BlackScholes, "call", S0, K, T, r)  # binding the function
  sigmas = np.linspace(0.01, 10, 1000)
  
  plt.plot(sigmas, BS_sigma(sigmas))
  plt.xlabel("sig")
  plt.ylabel("price")
  plt.title("Black-Scholes price as function of volatility")
  plt.show()
  
