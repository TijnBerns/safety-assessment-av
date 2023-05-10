import sys
from typing import List, Tuple, Self, Union
import numpy as np
import scipy.stats

class Interval():
    def __init__(self, start: float, end:float) -> None:
        self.start = start
        self.end = end
        self.as_tuple = (start, end)
        
    def length(self) -> float:
        return np.abs(self.start - self.end)
    
    def in_interval(self, x: float) -> bool:
        return x >= self.start and x <= self.end
    
    def __repr__(self) -> str:
        return f"({self.start} {self.end})"
    
    def __lt__(self, other: Self) -> bool:
        return self.as_tuple < other.as_tuple
    
    def __getitem__(self, i: int):
        return self.as_tuple[i]
        
    
class MultiInterval():
    def __init__(self, intervals: List[Tuple[float, float]]) -> None:
        self.intervals = [Interval(*i) for i in intervals]
        self.as_tuples = intervals
        
    def union(self, other: Union[Self, List[Tuple[float, float]]]) -> Self:
        """Computes the intersection between two lists of intervals
        """
        if type(other) == MultiInterval:
            c = MultiInterval((*self.intervals, *other.intervals))
        else:
            c = MultiInterval((*self.intervals, *other))
            
        out = []
        for begin, end in sorted(c):
            if out and out[-1][1] >= begin:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([begin, end])
        return MultiInterval(out)

        
    def intersection(self, other: Union[Self, List[Tuple[float, float]]]) -> Self:
        """Computes the intersection between two lists of intervals
        """
        i, j = 0, 0
        n = len(self)
        m = len(other)
        out = []
        while i < n and j < m:
            l = max(self[i][0], other[j][0])
            r = min(self[i][1], other[j][1])

            if l <= r:
                out.append([l, r])

            if self[i][1] < other[j][1]:
                i += 1
            else:
                j += 1

        return MultiInterval(out)
    
    def in_interval(self, x: float) -> bool:
        """Returns whether x in the the list of intervals

        Args:
            x (float): Value to be checked for

        Returns:
            bool: True if input is in intervals, false otherwise
        """
        for i in self.intervals: 
            if i.in_interval(x):
                return True
        return False
    
    def jaccard_distance(self, other: Union[Self, List[Tuple[float, float]]], accept: float=1e-4, max_attempts: int= 400) -> float:
        """Computes the Jacobian distance between two sets of intervals in a Monte Carlo manner
        Returns:
            float: Jacobian distance between y and y_hat
        """
        intersection = self.intersection(other)
        union = self.union(other)
        return 1 - sum([i.length() for i in intersection]) / sum([i.length() for i in union])
    
    def min(self) -> float:
        return min(self.intervals).start
    
    def max(self) -> float:
        return sorted(self.intervals, key= lambda x: x[1])[-1].end
        

    def __repr__(self) -> str:
        return f"MultiInterval {self.intervals}"
    
    def __getitem__(self, i):
        return self.intervals[i]

    def __len__(self):
        return len(self.intervals)
    
    
if __name__== "__main__":
    a = MultiInterval([[0, 1], [3,4]])
    b = MultiInterval([[1,2],[3,5],[6,7]])
    c = a.intersection(b)
    d = a.union(b)
    print(c)
    print(d)
    print(c.in_interval(5))
    print(d.in_interval(5))
    breakpoint()
        
    
    