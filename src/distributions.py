from abc import ABC
import scipy.stats
import numpy as np
from typing import List, Union
import utils
import matplotlib.pyplot as plt


class Distribution(ABC):
    def rvs(self, n: int):
        pass
    
    def pdf(self, x: Union[np.ndarray, float]):
        pass
    
class Mixture(Distribution):
    def __init__(self, weights: List[float], distributions: List[Distribution]) -> None:
        super().__init__()
        assert sum(weights) == 1
        assert len(weights) == len(distributions)
        
        self.weights = np.sort(weights)
        self.distributions = distributions
        
    def rvs(self, n: int) -> np.ndarray:
        res = np.array([])
        for w, dist in zip(self.weights, self.distributions):
            r = np.random.rand(2*n)
            samples = dist.rvs(2*n)
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
        
    
    
class UnivariateThreshold(Distribution):
    def __init__(self, distribution, threshold_distribution) -> None:
        super().__init__()
        self.distribution = distribution
        self.threshold_distribution = threshold_distribution
        
    def rvs(self, n):
        x = np.sort(self.distribution.rvs(n))
        y = np.sort(self.threshold_distribution.rvs(n)) + scipy.stats.norm().rvs(n)
        
        res = np.stack((x, y), axis=1)
        np.random.shuffle(res)
        
        return res 
    
    def pdf(self, x):
        return self.distribution.pdf(x)

    
class BivariatePoison(Distribution):
    def __init__(self, mean) -> None:
        self.mean = mean
        self.p1 = scipy.stats.poisson(mean[0])
        self.p2 = scipy.stats.poisson(mean[1])
        self.p3 = scipy.stats.poisson(mean[2])
    
    def rvs(self, n):
        y1 = self.p1.rvs(n)
        y2 = self.p1.rvs(n)
        y3 = self.p1.rvs(n)
        
        x1 = y1 + y3
        x2 = y2 + y3

        return np.stack((x1, x2), axis=1)
    
    def pdf(self, x): 
        pass
        
if __name__ =="__main__":
    # rv = UnivariateThreshold(scipy.stats.cauchy(), scipy.stats.norm())
    rv = Mixture([0.7,0.3], [scipy.stats.norm(-2, 1), scipy.stats.norm(2, 1)])
    x = np.linspace(-8,8,1000)

    _, ax = plt.subplots(1,1)
    ax.hist(rv.rvs(10_000),bins=100, density=True,  alpha=0.5, color="tab:blue")
    ax.plot(x, rv.pdf(x), color="tab:blue")
    plt.show()
    