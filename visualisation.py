from kernelFuncts import RadialBasisFunction, KernelFunction, RationalQuadraticFunction, PeriodicFunction, CompositeKernel, LinearFunction
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualiseKernel(kernel:KernelFunction):
    """This function visualises the kernel."""
    matrixOfValues = kernel(np.linspace(0,5, 50))
    values = kernel(np.array([0]), np.linspace(-5,5,100))

    fig, axes = plt.subplots(1,2, width_ratios=[1,1], height_ratios=[1])

    # first axis is graph of values
    axes[0].plot(np.linspace(-5,5,100), values)
    axes[0].set_box_aspect(1)
    axes[0].set_xlabel("(x-y)")
    axes[0].set_ylabel("Kernel output")
    
    # second axis is visual pattern
    im = axes[1].imshow(matrixOfValues)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # fig.suptitle(f"σ = {kernel.sigma} Lengthscsale = {kernel.lengthscale}", fontsize=16)
    fig.tight_layout()
    fig.savefig(f"{kernel.kernType}.png")
    # plt.show()


def showAll():
    rbf = RadialBasisFunction(lengthscale=1)
    rq = RationalQuadraticFunction(alpha=1)
    p = PeriodicFunction()
    lin = LinearFunction()
    comp = CompositeKernel(p, RadialBasisFunction(lengthscale=5), "x")
    comp.kernType = "Locally Periodic"
    kernels = [rbf, rq, p, comp]

    fig, axes = plt.subplots(2, len(kernels), figsize = (18,8))
    for i, kernel in enumerate(kernels):
        matrixOfValues = kernel(np.linspace(0,5, 50))
        values = kernel(np.array([0]), np.linspace(-5,5,100))
        axes[0][i].plot(np.linspace(-5,5,100), values)
        axes[0][i].set_box_aspect(1)
        axes[0][i].set_xlabel("(x - y)", fontsize = 12, fontfamily = "serif")

        
        # second axis is visual pattern
        im = axes[1][i].imshow(matrixOfValues)
        axes[1][i].set_xticks([])
        axes[1][i].set_yticks([])
        divider = make_axes_locatable(axes[1][i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        axes[0][i].set_title(kernel.kernType, fontsize = 18, fontfamily = "serif")

    for i in range(1,len(kernels)):
        axes[0][i].set_yticks([])
    axes[0][0].set_ylabel("Kernel output", fontsize = 12, fontfamily = "serif")

    fig.savefig("Comparisons.png")    


def testHypes():
    
    kernels = [RadialBasisFunction(1,1), RadialBasisFunction(1,2), RadialBasisFunction(0.75,1), RadialBasisFunction(0.75,2)]

    fig, axes = plt.subplots(2, len(kernels), figsize = (18,8))
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

        axes[0][i].set_title(f"Sigma: {kernel.sigma}, lengthscale: {kernel.lengthscale}", fontsize = 15, fontfamily = "serif")

    for i in range(1,len(kernels)):
        axes[0][i].set_yticks([])
    axes[0][0].set_ylabel("Kernel output", fontsize = 12, fontfamily = "serif")

    fig.savefig("hyperparams.png")   

if __name__ == "__main__":
    # rbf = RadialBasisFunction(lengthscale=5)
    # rq = RationalQuadraticFunction(alpha=1)
    # p = PeriodicFunction()
    # lin = LinearFunction()
    # comp = CompositeKernel(p, rbf, "x")
    # visualiseKernel(lin)
    # showAll()
    testHypes()