from abc import abstractmethod, ABC
import numpy as np
from typing import Union

class KernelFunction(ABC):
    """Abstract kernel class.
    Attributes:
        kernType (str): the type of kernel.
    """
    kernType = ""
    @abstractmethod
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray, y: Union[np.ndarray, None] = None) -> np.ndarray:
        """Calling this kernel class will calculate the kernel values.
        Args:
            x: first numpy array.
            y: second numpy array. Default None.
        """
        if y is None:
            y = x
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        return self.kernel(x,y)
        
    @abstractmethod
    def kernel(self, x: np.ndarray, y:np.ndarray) -> np.array:
        return np.array([0])
    

class RadialBasisFunction(KernelFunction):
    """Radial basis function.
    Attributes:
        kernType (str): the type of kernel.
        sigma (float): the variance of the kernel.
        lengthscale (float): the lengthscale of the kernel.
    """
    def __init__(self, sigma: float = 1.0, lengthscale: float = 1.0):
        self.kernType = "Radial Basis"
        self.sigma = sigma
        self.lengthscale = lengthscale
        
    
    def kernel(self, x: np.ndarray, y: np.ndarray):
        return self.sigma**2 * np.exp(- (x-y[:, None])**2/(2*self.lengthscale**2))
    

class RationalQuadraticFunction(KernelFunction):
    """RationalQuadraticFunction.
    Attributes:
        kernType (str): the type of kernel.
        sigma (float): the variance of the kernel.
        lengthscale (float): the lengthscale of the kernel.
        alpha (float): the exponential scaler.
    """
    def __init__(self, sigma: float = 1.0, lengthscale: float = 1.0, alpha: 1.0 = 1):
        self.kernType = "Rational Quadratic"
        self.sigma = sigma
        self.lengthscale = lengthscale
        self.alpha = alpha
        
    
    def kernel(self, x: np.ndarray, y: np.ndarray):
        return self.sigma**2 *(1+ ((x-y[:, None])**2)/(2*self.alpha * self.lengthscale**2))**(-self.alpha)
    


class PeriodicFunction(KernelFunction):
    """Periodic kernel.
    Attributes:
        kernType (str): the type of kernel.
        sigma (float): the variance of the kernel.
        lengthscale (float): the lengthscale of the kernel. 
    """
    def __init__(self, sigma: float = 1.0, lengthscale: float = 1.0, period: float = 2):
        self.kernType = "Periodic"
        self.sigma = sigma
        self.lengthscale = lengthscale
        self.period = period
        
    
    def kernel(self, x: np.ndarray, y: np.ndarray):
        return self.sigma**2 * np.exp(- 2*np.sin(np.pi*np.absolute(x-y[:, None])/self.period)**2/(2*self.lengthscale**2))
    


class CompositeKernel(KernelFunction):
    """Composite kernel defined by operations.
    This kernel is created from multiplying two kernels.
    Attributes:
        kernType (str): the type of kernel.
        kern1 (KernelFunction): the first kernel.
        kern2 (KernelFunction): the second kernel.
        operation (str): the operation.
    """
    def __init__(self,kern1: KernelFunction, kern2: KernelFunction, operation: str):
        assert(operation in ["+","-","x"])
        self.kernType = "Composite"
        self.kern1 = kern1
        self.kern2 = kern2
        self.operation = operation
        
    
    def kernel(self, x: np.ndarray, y: np.ndarray):
        operation = self.operation
        match operation:
            case "+":
                return self.kern1(x,y)+self.kern2(x,y)
            case "-":
                return self.kern1(x,y)-self.kern2(x,y)
            case "x":
                return self.kern1(x,y)*self.kern2(x,y)
            case _:
                return None
            
    
class LinearFunction(KernelFunction):
    """linear kernel.
    Attributes:
        sigmaA (float): Variance 1.
        sigmaV (float): Variance 2.
        C (float): shrinkage. 
    """

    def __init__(self, sigmaA: float = 0.0, sigmaV: float = 1.0, C: float = -1.0):
        self.kernType = "Linear"
        self.sigmaA = sigmaA
        self.sigmaV = sigmaV
        self.C = C        
    
    def kernel(self, x: np.ndarray, y: np.ndarray):
        return self.sigmaA + self.sigmaV*(x-self.C)*(y[:,None]-self.C)
    