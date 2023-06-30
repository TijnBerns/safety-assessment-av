import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
import numpy as np


import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "8"

FIGSIZE = (3.0, 2.0)
# DATA = np.sort(norm().rvs(10, random_state=10))
DATA1 = [-2.1, 	-1.3, 	-0.4, 	1.9, 	5.1, 	6.2]
DATA2 = np.sort(norm().rvs(100, random_state=10))


def plot():
    x_values = np.linspace(-6,11, 1000)
    fig, axs = plt.subplots(1,1, figsize=FIGSIZE)
    axs.set_xlabel('$x$')
    axs.set_ylabel('density')
    res = np.zeros_like(x_values)
    for mean in DATA1:
        pdf = norm().pdf(x_values - mean) / len(DATA1)
        res += pdf
        axs.plot(x_values, pdf, color='tab:orange', linestyle='--')
   
    axs.plot(x_values, res, color='tab:blue')        
    plt.tight_layout()
    plt.savefig('img/kde_components.pgf')
    
    x_values = np.linspace(-4,4, 10000)
    fig, axs = plt.subplots(1,1, figsize=FIGSIZE)
    axs.set_xlabel('$x$')
    axs.set_ylabel('density')
    # axs.plot(x_values, norm().pdf(x_values), label='true')
    H = [0.08,0.32,1.28]
    for h in H:
        kde = gaussian_kde(dataset=DATA2, bw_method=h)
        pdf = kde.pdf(x_values)
        axs.plot(x_values, pdf, label=f'$h={h}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('img/kde_bw.pgf')
    

if __name__  == '__main__':
    plot()