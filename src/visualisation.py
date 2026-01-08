from src.kernelFuncts import RadialBasisFunction, KernelFunction, RationalQuadraticFunction, PeriodicFunction, CompositeKernel, LinearFunction
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.GaussianProcess import GaussianProcess
from src.data import generateLinearData, generateSinData

alphabet = "abcdefghi"

def visualiseKernel(kernel:KernelFunction):
    """This function visualises the kernel."""
    matrixOfValues = kernel(np.linspace(0,5, 50))
    x = np.linspace(-5,5,1000)
    values = kernel(np.array([0]), x)

    fig, axes = plt.subplots(1,2, width_ratios=[1,1], height_ratios=[1], figsize = (18,8))

    # first axis is graph of values
    axes[0].plot(np.linspace(-5,5,1000), values)
    axes[0].set_box_aspect(1)
    axes[0].set_xlabel("(x-y)", fontsize = 20, fontfamily = "serif")
    axes[0].set_ylabel("Kernel output", fontsize = 20, fontfamily = "serif")
    
    # second axis is visual pattern
    im = axes[1].imshow(matrixOfValues, cmap = "magma", vmin = 0, vmax = 1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel("x", fontsize = 20, fontfamily = "serif")
    axes[1].set_ylabel("x'   ", fontsize = 20, fontfamily = "serif", rotation=0)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=20) 
    for index, axis in enumerate(axes):
        fixTickParams(axis, index)
    axes[0].set_xlim(np.min(x),np.max(x))

    fig.tight_layout()
    fig.savefig(f"figures/{kernel.kernType.replace(" ", "")}.png")
    # plt.show()


def showAll():
    rbf = RadialBasisFunction(lengthscale=1)
    rq = RationalQuadraticFunction(alpha=1)
    p = PeriodicFunction()
    lin = LinearFunction()
    comp = CompositeKernel(p, RadialBasisFunction(lengthscale=5), "x")
    comp.kernType = "Locally Periodic"
    kernels = [rbf, rq, p, comp]
    x = np.linspace(-5,5,100)
    fig, axes = plt.subplots(2, len(kernels), figsize = (18,8))
    for i, kernel in enumerate(kernels):
        matrixOfValues = kernel(np.linspace(0,5, 50))
        values = kernel(np.array([0]), x)
        axes[0][i].plot(x, values)
        axes[0][i].set_box_aspect(1)
        # axes[0][i].set_xlabel("(x - y)", fontsize = 12, fontfamily = "serif")

        
        # second axis is visual pattern
        im = axes[1][i].imshow(matrixOfValues, cmap = "magma", vmin = 0, vmax = 1)
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])
        divider = make_axes_locatable(axes[1][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


    for i in range(1,len(kernels)):
        axes[0][i].set_yticks([])
        axes[0][i].set_xlim(np.min(x),np.max(x))

    axes[0][0].set_ylabel("Kernel output", fontsize = 12, fontfamily = "serif")

    for index, axis in enumerate(axes.flatten()):
        fixTickParams(axis, index)

    fig.savefig("figures/Comparisons.png")    


def testHypes():
    
    kernels = [RadialBasisFunction(1,1), RadialBasisFunction(1,2), RadialBasisFunction(0.75,1), RadialBasisFunction(0.75,2)]
    x = np.linspace(-5,5,100)
    fig, axes = plt.subplots(2, len(kernels), figsize = (18,8))
    for i, kernel in enumerate(kernels):
        matrixOfValues = kernel(np.linspace(0,5, 50))
        values = kernel(np.array([0]), x)
        axes[0][i].plot(x, values)
        axes[0][i].set_box_aspect(1)
        axes[0][i].set_xlabel("(x - y)", fontsize = 12, fontfamily = "serif")
        axes[0][i].set_ylim([0,1.1])
        
        # second axis is visual pattern
        im = axes[1][i].imshow(matrixOfValues, vmin = 0, vmax = 1, cmap = "magma")
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])
        divider = make_axes_locatable(axes[1][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


    for i in range(1,len(kernels)):
        axes[0][i].set_yticks([])
        axes[0][i].set_xlim(np.min(x),np.max(x))

    axes[0][0].set_ylabel("Kernel output", fontsize = 12, fontfamily = "serif")
    for index, axis in enumerate(axes.flatten()):
        fixTickParams(axis, index)
    fig.savefig("figures/hyperparams.png")   


def priorDistSample(gp: GaussianProcess, nsamples: int = 3, fname="figures/priorVsPosterior.png"):
    """This function plots and samples functions from the means and covariance.
    Args:
        gp: the gaussian process having observed data.
    """

    fig, axes = plt.subplots(ncols=2, figsize = (18,6.3), sharey=True)
    gp.observeData(np.array([2,3, 7, 5]), np.array([0.5,0.5,2,1]))

    # calculate the mean and covariance for prior.
    
    kernel = gp.kernelFunction
    x = np.linspace(0,10,200)
    means = np.zeros(200)
    kernelMatrix = kernel(x)
    axes[0].plot(x,means)
    axes[0].fill_between(x, means-kernelMatrix.diagonal(), means+kernelMatrix.diagonal(), alpha = 0.1)
    postMeans,PostVariance = gp.inference(x)
    axes[1].plot(x,postMeans, label="Mean function")
    axes[1].fill_between(x, postMeans-PostVariance.diagonal(), postMeans+PostVariance.diagonal(), alpha = 0.1, label="Variance range")
    axes[1].scatter(gp.x, gp.y, marker = "x", c="black", s=40, label="Observed data")
    for i in range(nsamples):
        f = np.random.multivariate_normal(means, kernelMatrix)
        axes[0].plot(x, f, linestyle = "dashed", linewidth = 1)
        f = np.random.multivariate_normal(postMeans, PostVariance)
        axes[1].plot(x, f, linestyle = "dashed", linewidth = 1)

    # fig.legend(loc="center right")
    for index, axis in enumerate(axes):
        fixTickParams(axis, index)
        axis.set_yticks([-2,0,2])
        axis.set_ylim(-3.5,3.5)
        axis.set_xlim(np.min(x),np.max(x))
        axis.set_xticks([])
        axis.set_xlabel("x", fontsize = 20, fontfamily = "serif")

    axes[0].set_ylabel("f(x)", fontsize = 20, fontfamily = "serif")


    fig.tight_layout()

    fig.savefig(fname)


def GPComps(gps: list[GaussianProcess], fname:str, nsamples = 3):
    """Generates prior and posterior distribution according to multiple GPs."""
    observedX = [2,3,7, 5]
    observedY = [0.5,0.5,2,1]
    fig1, axes1 = plt.subplots(ncols=len(gps), figsize = (18,18/(len(gps)+1)), sharey=True)
    fig2, axes2 = plt.subplots(ncols=len(gps), figsize = (18,18/(len(gps)+1)), sharey=True)
    for index, gp in enumerate(gps):
        gp.observeData(observedX, observedY)
        kernel = gp.kernelFunction
        x = np.linspace(0,10,300)
        means = np.zeros(300)
        kernelMatrix = kernel(x)
        postMeans,PostVariance = gp.inference(x)
        axes1[index].plot(x,means)
        axes1[index].fill_between(x, means-kernelMatrix.diagonal(), means+kernelMatrix.diagonal(), alpha = 0.1)
        axes2[index].plot(x,postMeans, label="Mean function")
        axes2[index].fill_between(x, postMeans-PostVariance.diagonal(), postMeans+PostVariance.diagonal(), alpha = 0.1, label="Variance range")
        axes2[index].scatter(gp.x, gp.y, marker = "x", c="black", s=40, label="Observed data")
        for i in range(nsamples):
            f = np.random.multivariate_normal(means, kernelMatrix)
            axes1[index].plot(x, f, linestyle = "dashed", linewidth = 1)
            f = np.random.multivariate_normal(postMeans, PostVariance)
            axes2[index].plot(x, f, linestyle = "dashed", linewidth = 1)
    
    for axis in axes1:
        axis.tick_params(direction = "in", labelsize=15)
        axis.set_ylim(-3.5,3.5)
        axis.set_yticks([-2,0,2])
        axis.set_xticks([])
        axis.set_xlim(np.min(x),np.max(x))
        axis.set_xlabel("x", fontsize = 20, fontfamily = "serif")
    axes1[0].set_ylabel("f(x)", fontsize = 20, fontfamily = "serif")

    for axis in axes2:
        axis.tick_params(direction = "in", labelsize=15)
        axis.set_ylim(-3.5,3.5)
        axis.set_yticks([-2,0,2])
        axis.set_xticks([])
        axis.set_xlim(np.min(x),np.max(x))
        axis.set_xlabel("x", fontsize = 20, fontfamily = "serif")
    axes2[0].set_ylabel("f(x)", fontsize = 20, fontfamily = "serif")

    fig1.tight_layout()
    fig2.tight_layout()

    fig1.savefig(f"figures/{fname}Priors.png")
    fig2.savefig(f"figures/{fname}Posteriors.png")

    
def visualiseDistribution(gp: GaussianProcess):
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize = (18,12), sharey=True)

    # create a big axis for the GP posterior
    gs = axes[0,0].get_gridspec()
    for axis in axes[0]:
        axis.remove()
    axis = fig.add_subplot(gs[0, :])

    # plot the GP posterior
    observedX = [2,3,6, 4.8]
    observedY = [-0.5,0.5,2,1]
    gp.observeData(observedX, observedY)

    inspectionPoints = [1.5,5.0,8.5]

    x = np.linspace(0,10,200)
    means, kernelMatrix = gp.inference(x)
    axis.plot(x,means)
    axis.scatter(gp.x, gp.y, marker = "x", c="black", s=40, label="Observed data")
    axis.fill_between(x, means-kernelMatrix.diagonal(), means+kernelMatrix.diagonal(), alpha = 0.1)
    axis.tick_params(direction = "in", labelsize=15)
    axis.set_ylim(-3.5,3.5)
    axis.set_ylabel("f(x)", fontsize = 20, fontfamily = "serif")
    axis.set_xlabel("x", fontsize = 20, fontfamily = "serif")
    axis.set_yticks([-2,0,2])
    axis.set_xticks(inspectionPoints)
    axis.set_xlim(np.min(x),np.max(x))
    fixTickParams(axis,0)

    for index, inspectionPoint in enumerate(inspectionPoints):
        axis.plot([inspectionPoint, inspectionPoint], [-4, 4], color = "black",linestyle = "dashed")
        mean, var = gp.inference(inspectionPoint)
        x = np.linspace(mean-4,mean+4,200)
        y = pdf(x, mean, var)
        axes[1,index].plot(x,y.flatten())
        axes[1,index].fill_between(x.flatten(),y.flatten(), 0, alpha=0.1)
        axes[1,index].set_xticks([-4,-2,0,2,4])
        axes[1,index].set_xlim([np.min(x), np.max(x)])
        axes[1,index].set_xlabel(f"f({inspectionPoint})", fontsize = 20, fontfamily = "serif")
        fixTickParams(axes[1,index], index)


    axes[1,0].set_ylabel("Probablity density", fontsize = 20, fontfamily = "serif")

    axes[1,0].set_ylim(0,3)
    axes[1,0].set_yticks([0,1,2,3])
    # fig.tight_layout()
    fig.savefig("figures/distribution.png")

def pdf(x: Union[float,np.ndarray], mean: float, variance: float):
    """Function provides probability density function for x."""
    std = np.sqrt(variance)
    y_out = 1/(std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
    return y_out


def fixTickParams(axis, index):
    """This function fixes axis for consistency."""
    axis.tick_params(direction = "in", labelsize=15)
    # axis.set_title(f"({alphabet[index]})", size = 30, fontfamily = "serif", fontweight = "bold")

    


if __name__ == "__main__":
    rbf = RadialBasisFunction(lengthscale=1)
    # rq = RationalQuadraticFunction(alpha=1)
    # p = PeriodicFunction()
    # lin = LinearFunction(sigmaV=1/36)
    # comp = CompositeKernel(p, RadialBasisFunction(lengthscale=4), "x")
    # visualiseKernel(rbf)
    # visualiseKernel(rq)
    # visualiseKernel(p)
    # visualiseKernel(lin)
    # visualiseKernel(comp)
    # showAll()
    # testHypes()
    # showGPLengthscales()
    gp = GaussianProcess(RadialBasisFunction(lengthscale=1, sigma=1), noise_variance=0.01)
    priorDistSample(gp, nsamples=5)
    gps = [
        GaussianProcess(RadialBasisFunction(), noise_variance=0.01), GaussianProcess(RadialBasisFunction(), noise_variance=0.1), GaussianProcess(RadialBasisFunction(), noise_variance=1)
    ]
    GPComps(gps, "noiseVariance", nsamples=4)
    gps = [
        GaussianProcess(RadialBasisFunction(sigma=0.5)), GaussianProcess(RadialBasisFunction(sigma=1)), GaussianProcess(RadialBasisFunction(sigma=1.4))
    ]
    GPComps(gps, "sigmas", nsamples=4)
    gps = [
        GaussianProcess(RadialBasisFunction(lengthscale=0.1)), GaussianProcess(RadialBasisFunction(lengthscale=0.5)), GaussianProcess(RadialBasisFunction(lengthscale=2))
    ]
    GPComps(gps, "lengthscales", nsamples=3)
    visualiseDistribution(GaussianProcess(rbf))