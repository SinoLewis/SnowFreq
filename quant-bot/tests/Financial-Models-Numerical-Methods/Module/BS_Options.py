from FMNM.BS_pricer import BS_pricer
from FMNM.Parameters import Option_param
from FMNM.Processes import Diffusion_process

def bs_options(S0=100, K=100, T=1, r=0.1, sig=0.2, exercise="European", payoff="call"):
    # Creates the object with the parameters of the option
    opt_param = Option_param(S0, K, T, exercise, payoff)
    # Creates the object with the parameters of the process
    diff_param = Diffusion_process(r, sig)
    # Creates the pricer object
    BS = BS_pricer(opt_param, diff_param)

    # BS.closed_formula()
    # BS.BS.Fourier_inversion()
    return BS