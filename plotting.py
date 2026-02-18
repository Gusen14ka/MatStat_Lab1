import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def plot_distributions(sample, name, n):
    plt.figure()
    
    plt.grid()

    

    if(name == "poisson"):
        bins = np.arange(min(sample), max(sample)+2)
        plt.hist(sample, bins, density=True, alpha=0.67, edgecolor="black", color="skyblue", label="Генерация")
        values = np.arange(min(sample) - int(min(sample) * 0.2), max(sample) + 1 + int(min(sample) * 0.2))
        probs = st.poisson.pmf(values, 5)
        plt.xlim(-25,25)
        plt.plot(values, probs, '-o', color="red", label="Теория")
    
    else:
        plt.hist(sample, bins="fd", density=True, alpha=0.67, edgecolor="black", color="skyblue", label="Генерация")
        data_min = np.min(sample)
        data_max = np.max(sample)
        span = data_max - data_min
        padding = span * 0.2 if span > 0 else 1.0

        if name == "cauchy":
            x_min, x_max = -25, 25
        elif name == "uniform":
            loc = -np.sqrt(3)
            scale = 2 * np.sqrt(3)
            x_min = loc - padding
            x_max = loc + scale + padding
        else:
            x_min = data_min - padding
            x_max = data_max + padding
            x_min = min(x_min, -5)
            x_max = max(x_max, 5)

        x = np.linspace(x_min, x_max, 10000)

        pdf_functions = {
            "normal": lambda x: st.norm.pdf(x, 0, 1),
            "cauchy": lambda x: st.cauchy.pdf(x, 0, 1),
            "laplace": lambda x: st.laplace.pdf(x, 0, np.sqrt(1/2)),
            "uniform": lambda x: st.uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3))
        }

        y = pdf_functions[name](x)
        plt.xlim(x_min, x_max)
        plt.plot(x, y, color="red", label="Теория")

    plt.title(f"Распределение {name}, n={n}")
    plt.xlabel("Значение случайно величины")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    