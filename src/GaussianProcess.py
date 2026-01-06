"""This file is for the Gaussain process."""
from src.kernelFuncts import KernelFunction, RadialBasisFunction
from src.data import generateLinearData, generateSinData
import numpy as np
from typing import Union
import matplotlib.pyplot as plt


class GaussianProcess():
    def __init__(self, kernelFunction: KernelFunction, noise_variance = 1):
        self.kernelFunction = kernelFunction
        self.x = None
        self.y = None
        self.kAA = np.atleast_1d(noise_variance)
        self.noise_variance = noise_variance
        

    def observeData(self, x: Union[int, np.ndarray], y: Union[int, np.ndarray]):
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.kAA = self.kernelFunction(x)

    def inference(self, x: Union[int, np.ndarray]):
        x = np.atleast_1d(x)
        kAB = self.kernelFunction(self.x, x)
        kBB = self.kernelFunction(x)
        kAAp = self.kAA + np.eye(len(self.x))*self.noise_variance

        L = np.linalg.cholesky(kAAp)

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))

        mu = kAB @ alpha

        v = np.linalg.solve(L, kAB.T)
        sigma = kBB - v.T @ v

        return mu, sigma

    def meanFunction(self,x: Union[int, np.ndarray]):
        return 0
    
    def visualiseFit(self):
        means, variances = self.inference(self.x)
        fig = plt.figure(figsize= (20,15))
        plt.scatter(self.x,self.y, s=3, marker="x", c="black")
        plt.plot(self.x, means)
        plt.fill_between(self.x, means-np.diagonal(variances), means+np.diagonal(variances), alpha = 0.1, color = "b")
        plt.savefig("figures/fitVis.png")
        
    

if __name__ == "__main__":
    gp = GaussianProcess(RadialBasisFunction(sigma=1, lengthscale=10))
    x = np.linspace(0,20, 1000)
    y = generateSinData(x, sigma=0.5)
    gp.observeData(x,y)
    # print(gp.inference(5))
    gp.visualiseFit()