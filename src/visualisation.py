from src.kernelFuncts import RadialBasisFunction, KernelFunction, RationalQuadraticFunction, PeriodicFunction, CompositeKernel, LinearFunction
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.GaussianProcess import GaussianProcess
from src.data import generateLinearData, generateSinData

alphabet = "abcdefghi"

def visualiseKernel(kernel:KernelFunction):
    """This function visualises the kernel."""
    matrixOfValues = kernel(np.linspace(0,5, 50))
    values = kernel(np.array([0]), np.linspace(-5,5,1000))

    fig, axes = plt.subplots(1,2, width_ratios=[1,1], height_ratios=[1], figsize = (20,8))

    # first axis is graph of values
    axes[0].plot(np.linspace(-5,5,1000), values)
    axes[0].set_box_aspect(1)
    axes[0].set_xlabel("(x-y)", fontsize = 20, fontfamily = "serif")
    axes[0].set_ylabel("Kernel output", fontsize = 20, fontfamily = "serif")
    
    # second axis is visual pattern
    im = axes[1].imshow(matrixOfValues, cmap = "magma", vmin = 0, vmax = 1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=20) 
    for index, axis in enumerate(axes):
        fixTickParams(axis, index)

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

    fig, axes = plt.subplots(2, len(kernels), figsize = (20,8))
    for i, kernel in enumerate(kernels):
        matrixOfValues = kernel(np.linspace(0,5, 50))
        values = kernel(np.array([0]), np.linspace(-5,5,100))
        axes[0][i].plot(np.linspace(-5,5,100), values)
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
    axes[0][0].set_ylabel("Kernel output", fontsize = 12, fontfamily = "serif")

    for index, axis in enumerate(axes.flatten()):
        fixTickParams(axis, index)

    fig.savefig("figures/Comparisons.png")    


def testHypes():
    
    kernels = [RadialBasisFunction(1,1), RadialBasisFunction(1,2), RadialBasisFunction(0.75,1), RadialBasisFunction(0.75,2)]

    fig, axes = plt.subplots(2, len(kernels), figsize = (20,8))
    for i, kernel in enumerate(kernels):
        matrixOfValues = kernel(np.linspace(0,5, 50))
        values = kernel(np.array([0]), np.linspace(-5,5,100))
        axes[0][i].plot(np.linspace(-5,5,100), values)
        axes[0][i].set_box_aspect(1)
        axes[0][i].set_xlabel("(x - y)", fontsize = 12, fontfamily = "serif")
        axes[0][i].set_ylim([0,1.1])
        
        # second axis is visual pattern
        im = axes[1][i].imshow(matrixOfValues, vmin = 0, vmax = 1)
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])
        divider = make_axes_locatable(axes[1][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


    for i in range(1,len(kernels)):
        axes[0][i].set_yticks([])
    axes[0][0].set_ylabel("Kernel output", fontsize = 12, fontfamily = "serif")
    for index, axis in enumerate(axes.flatten()):
        fixTickParams(axis, index)
    fig.savefig("figures/hyperparams.png")   

def showGPLengthscales():
    fig, axes = plt.subplots(1,3, figsize = (20,8))
    x = np.linspace(0,20,250)
    y = generateSinData(x, 5, 0.1)
    gps = [GaussianProcess(RadialBasisFunction(lengthscale=0.1)), GaussianProcess(RadialBasisFunction(lengthscale=1)), GaussianProcess(RadialBasisFunction(lengthscale=10))]
    for i in range(3):
        gps[i].observeData(x,y)
        means, variances = gps[i].inference(x)
        axes[i].scatter(x,y, s=4, marker="x", c="black")
        axes[i].plot(x, means)
        axes[i].fill_between(x, means-np.diagonal(variances)*2, means+np.diagonal(variances)*2, alpha = 0.1, color = "b")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for index, axis in enumerate(axes):
        fixTickParams(axis, index)
    fig.tight_layout()
    fig.savefig("figures/lengthscales.png")


def priorDistSample(gp: GaussianProcess, nsamples: int = 3):
    """This function plots and samples functions from the means and covariance.
    Args:
        gp: the gaussian process having observed data.
    """

    fig, axes = plt.subplots(ncols=2, figsize = (20,10), sharey=True)
    
    # calculate the mean and covariance for prior.
    sigma = gp.noise_variance
    
    kernel = gp.kernelFunction
    x = np.linspace(0,100,100)
    means = np.zeros(100)
    kernelMatrix = kernel(x)
    axes[0].plot(x,means)
    axes[0].fill_between(x, means-kernelMatrix.diagonal(), means+kernelMatrix.diagonal(), alpha = 0.1)
    postMeans,PostVariance = gp.inference(x)
    axes[1].plot(x,postMeans, label="Mean function")
    axes[1].fill_between(x, postMeans-PostVariance.diagonal(), postMeans+PostVariance.diagonal(), alpha = 0.1, label="Variance range")
    axes[1].scatter(gp.x, gp.y, marker = "x", c="black", s=40, label="Observed data")
    f = np.random.multivariate_normal(means, kernelMatrix)
    axes[0].plot(x, f, linestyle = "dashed", color = "black", linewidth = 1, label="Sample")
    f = np.random.multivariate_normal(postMeans, PostVariance)
    axes[1].plot(x, f, linestyle = "dashed", color = "black", linewidth = 1)

    for i in range(nsamples-1):
        f = np.random.multivariate_normal(means, kernelMatrix)
        axes[0].plot(x, f, linestyle = "dashed", color = "black", linewidth = 1)
        f = np.random.multivariate_normal(postMeans, PostVariance)
        axes[1].plot(x, f, linestyle = "dashed", color = "black", linewidth = 1)

    # fig.legend(loc="center right")
    for index, axis in enumerate(axes):
        fixTickParams(axis, index)
    fig.savefig("figures/priorVsPosterior.png")


    
def fixTickParams(axis, index):
    """This function fixes axis for consistency."""
    axis.tick_params(direction = "in", labelsize=15)
    axis.set_title(f"({alphabet[index]})", size = 30, fontfamily = "serif", fontweight = "bold")

    


if __name__ == "__main__":
    rbf = RadialBasisFunction(lengthscale=1)
    rq = RationalQuadraticFunction(alpha=1)
    p = PeriodicFunction()
    lin = LinearFunction()
    comp = CompositeKernel(p, RadialBasisFunction(lengthscale=4), "x")
    visualiseKernel(rbf)
    visualiseKernel(rq)
    visualiseKernel(p)
    visualiseKernel(lin)
    visualiseKernel(comp)
    showAll()
    testHypes()
    showGPLengthscales()
    gp = GaussianProcess(RadialBasisFunction(lengthscale=5, sigma=1), noise_variance=0.01)
    gp.observeData(np.array([20,30,70, 50]), np.array([0.5,0.5,2,1]))
    priorDistSample(gp)