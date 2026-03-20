import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from visualisation import pdf
def generate_data(n: int, distribution: str = "normal"):
    y = np.random.normal(0, 1, n)
    return y

def compare_imps():
    y = generate_data(10000)

    # create mask
    random_mask = np.full(10000, True)
    random_mask[:3000] = False
    np.random.shuffle(random_mask)

    y_masked = y[random_mask]
    mean_masked = np.mean(y_masked)

    # impute data
    y_imputed = copy(y)
    for i, _ in enumerate(y_imputed):
        if not random_mask[i]:
            y_imputed[i] = mean_masked

    print(np.sum(y-y_imputed))

    # now plot
    var = np.var(y)
    var_masked = np.var(y_masked)
    var_imputed = np.var(y_imputed)

    print(var, var_masked, var_imputed)

    fig, axes = plt.subplots(ncols=3, figsize = (18,4.5), sharey = True)

    x = np.linspace(-4,4,200)
    dist = pdf(x, np.mean(y), var)
    dist_masked = pdf(x, mean_masked, var_masked)
    dist_imputed = pdf(x, np.mean(y_imputed), var_imputed)

    axes[0].hist(y,15, density=True, rwidth = 0.8, range = [-4, 4])
    axes[1].hist(y_masked, 15, density=True, rwidth = 0.8, range = [-4, 4])
    axes[2].hist(y_imputed, 15, density=True, rwidth = 0.8, range = [-4, 4])

    axes[0].plot(x, dist.flatten())
    axes[1].plot(x, dist_masked.flatten())
    axes[2].plot(x, dist_imputed.flatten())
    axes[0].set_yticks([0, 0.5, 1])
    axes[0].set_ylabel("Probability Density", fontsize = 16, fontfamily = "serif")
    for axis in axes:
        axis.set_xlabel("x", fontsize = 16, fontfamily = "serif")
    axes[0].set_title('Full Sample', size = 20, fontfamily = "serif")
    axes[1].set_title('70% of Sample', size = 20, fontfamily = "serif")
    axes[2].set_title('Imputed Sample', size = 20, fontfamily = "serif")

    fig.tight_layout()
    fig.savefig("figures/imputeComp.png")


if __name__ == "__main__":
    compare_imps()