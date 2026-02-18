from distributions import *
from plotting import plot_distributions
import matplotlib.pyplot as plt

def main():
    sample_size = [10, 100, 1000]

    distribustions = {
        "normal": generate_normal,
        "cauchy": generate_cauchy,
        "laplace": generate_laplace,
        "poisson": generate_poisson,
        "uniform": generate_uniform
    }

    for n in sample_size:
        for name, gen in distribustions.items():
            sample = gen(n)
            plot_distributions(sample, name, n)

    plt.show()

if __name__ == "__main__":
    main()