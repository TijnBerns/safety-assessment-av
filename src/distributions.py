from abc import ABC
import scipy.stats
import numpy as np
from typing import List, Union
import utils
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, rv_histogram, norm, beta, multivariate_normal
from typing import Any

class Distribution(ABC):
    def rvs(self, n: int, random_state: Any = None):
        pass
    
    def pdf(self, x: Union[np.ndarray, float]):
        pass
    
    def ppf(self, x: Union[np.ndarray, float]):
        pass
    
    
class Gaussian_Copula(Distribution):
    def __init__(self, cor: np.ndarray, distributions: List[Distribution]) -> None:
        super().__init__()
        self.cor = cor
        self.distributions = distributions
    
    def rvs(self,  n: int, random_state: Any = None):
        
        r0 = [0] * self.cor.shape[0]
        mv_norm = multivariate_normal(mean=r0, cov=self.cor)
        rand_Nnorm = mv_norm.rvs(n, random_state=random_state)
        
        # step 2: convert the r * N multivariate variates to scores
        rand_Unorm = norm.cdf(rand_Nnorm)
        
        # step 3: draw N random variates for each of the marginal distributions
        rand_Nmar = np.zeros_like(rand_Unorm)
        for i, distribution in enumerate(self.distributions):
            rand_Nmar[:,i] = distribution.ppf(rand_Unorm[:,i])
            
        return rand_Nmar
        
    
class Mixture(Distribution):
    def __init__(self, weights: List[float], distributions: List[Distribution]) -> None:
        super().__init__()
        assert sum(weights) == 1
        assert len(weights) == len(distributions)
        
        self.weights = np.sort(weights)
        self.distributions = distributions
        
    def rvs(self, n: int, random_state: Any = None) -> np.ndarray:
        res = np.array([])
        for w, dist in zip(self.weights, self.distributions):
            r = np.random.rand(2*n)
            samples = dist.rvs(2*n, )
            samples = samples[r <= w]
            res = np.concatenate((res, samples)) if res.size else samples
        
        np.random.shuffle(res)
        res = res[:n]
        assert len(res) == n                    
        return res 
            
        
    
    def pdf(self, x: Union[np.ndarray, float]) -> np.ndarray:
        res = np.zeros(len(x))
        for w, dist in zip(self.weights, self.distributions):

            res = np.add(res, w * dist.pdf(x))
            
        return res
    
    def ppf(self, x):
        raise Exception("Incorrect implementation of ppf function")
        res = np.zeros(len(x))
        for w, dist in zip(self.weights, self.distributions):
            res = np.add(res, w * dist.ppf(x))
            
        return res
    
    def cdf(self, x):
        res = np.zeros(len(x))
        for w, dist in zip(self.weights, self.distributions):
            res = np.add(res, w * dist.cdf(x))
            
        return res
        
    
    
# class UnivariateThreshold(Distribution):
#     def __init__(self, distribution, threshold_distribution) -> None:
#         super().__init__()
#         self.distribution = distribution
#         self.threshold_distribution = threshold_distribution
        
#     def rvs(self, n):
#         x = np.sort(self.distribution.rvs(n))
#         y = np.sort(self.threshold_distribution.rvs(n)) + scipy.stats.norm().rvs(n)
        
#         res = np.stack((x, y), axis=1)
#         np.random.shuffle(res)
        
#         return res 
    
#     def pdf(self, x):
#         return self.distribution.pdf(x)

    
# class BivariatePoison(Distribution):
#     def __init__(self, mean) -> None:
#         self.mean = mean
#         self.p1 = scipy.stats.poisson(mean[0])
#         self.p2 = scipy.stats.poisson(mean[1])
#         self.p3 = scipy.stats.poisson(mean[2])
    
#     def rvs(self, n):
#         y1 = self.p1.rvs(n)
#         y2 = self.p1.rvs(n)
#         y3 = self.p1.rvs(n)
        
#         x1 = y1 + y3
#         x2 = y2 + y3

#         return np.stack((x1, x2), axis=1)
    
#     def pdf(self, x): 
#         pass
        
if __name__ =="__main__":
    # rv = UnivariateThreshold(scipy.stats.cauchy(), scipy.stats.norm())
    # rv = Mixture([0.7,0.3], [scipy.stats.norm(-2, 1), scipy.stats.norm(2, 1)])
    # x = np.linspace(-8,8,1000)

    # _, ax = plt.subplots(1,1)
    # ax.hist(rv.rvs(10_000),bins=100, density=True,  alpha=0.5, color="tab:blue")
    # ax.plot(x, rv.pdf(x), color="tab:blue")
    # plt.show()
    
    
    c_target = np.array([[  1.0, 0.8,],
                        [0.8,  1.0]])
    
    # rv1 = Mixture([0.7,0.3], [scipy.stats.norm(-2, 1), scipy.stats.norm(2, 1)])
    # rv = Gaussian_Copula(c_target, [rv1 , scipy.stats.norm()])
    rv1 = scipy.stats.beta(0.5, 0.5)
    rv = Gaussian_Copula(c_target, [scipy.stats.gumbel_r(), scipy.stats.norm()])
    
    x = np.linspace(-8,8,1000)
    _, ax = plt.subplots(1,2)
    y = rv.rvs(10_000)
    ax[0].hist(y[:,0],bins=100, density=True,  alpha=0.5, color="tab:blue")
    ax[0].plot(x, rv1.pdf(x), color="tab:blue")
    
#     # ax[0].hist(rv1.rvs(10_000),bins=100, density=True,  alpha=0.5, color="tab:blue")
#     print(np.corrcoef(y[:,0], y[:,1]))
#     plt.show()
    
    