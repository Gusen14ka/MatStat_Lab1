import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def plot_distributions(sample, name, n):
    
    # =========================
    # ======== POISSON ========
    # =========================
    if name == "poisson":
        plt.figure()
        plt.grid()

        # разумный диапазон
        lam = 5
        x_min = max(0, int(lam - 4*np.sqrt(lam)))
        x_max = int(lam + 4*np.sqrt(lam))
        values = np.arange(x_min, x_max + 1)

        # эмпирические частоты
        counts = np.array([np.sum(sample == k) for k in values])
        probs_emp = counts / len(sample)

        plt.bar(values, probs_emp, alpha=0.7, edgecolor="black", label="Генерация")

        # теоретические вероятности
        probs = st.poisson.pmf(values, lam)
        plt.plot(values, probs, 'ro-', label="Теория")

        plt.xlim(x_min - 1, x_max + 1)
        plt.title(f"Распределение {name}, n={n}")
        plt.xlabel("Значение случайной величины")
        plt.ylabel("Плотность вероятности")
        plt.legend()
        return


    # =========================
    # ========= CAUCHY ========
    # =========================
    if name == "cauchy":

        data_min = np.min(sample)
        data_max = np.max(sample)
        span = data_max - data_min
        padding = span * 0.1 if span > 0 else 1.0

        # ================= FULL RANGE =================
        plt.figure()
        plt.grid()

        plt.hist(sample, bins="fd", density=True,
                alpha=0.6, edgecolor="black", label="Генерация")

        x_full = np.linspace(data_min - padding,
                            data_max + padding, 10000)
        y_full = st.cauchy.pdf(x_full, 0, 1)

        plt.plot(x_full, y_full, color="red", label="Теория")
        plt.title(f"Распределение cauchy (полный диапазон), n={n}")
        plt.xlabel("Значение случайной величины")
        plt.ylabel("Плотность вероятности")
        plt.legend()


        # ================= CENTER =================
        plt.figure()
        plt.grid()

        plt.hist(sample, bins="fd", density=True,
                alpha=0.6, edgecolor="black", label="Генерация")

        # центр берём фиксированный, но если диапазон меньше — не расширяем
        center_limit = 25
        x_center = np.linspace(-center_limit, center_limit, 10000)
        y_center = st.cauchy.pdf(x_center, 0, 1)

        plt.plot(x_center, y_center, color="red", label="Теория")
        plt.xlim(-center_limit, center_limit)
        plt.title(f"Распределение cauchy (центр [-25,25]), n={n}")
        plt.xlabel("Значение случайной величины")
        plt.ylabel("Плотность вероятности")
        plt.legend()

        return


    # =========================
    # ======== ОСТАЛЬНЫЕ ======
    # =========================
    plt.figure()
    plt.grid()

    plt.hist(sample, bins="fd", density=True,
             alpha=0.6, edgecolor="black", label="Генерация")

    data_min = np.min(sample)
    data_max = np.max(sample)
    span = data_max - data_min
    padding = span * 0.2 if span > 0 else 1.0

    x_min = data_min - padding
    x_max = data_max + padding

    x = np.linspace(x_min, x_max, 10000)

    pdf_functions = {
        "normal": lambda x: st.norm.pdf(x, 0, 1),
        "laplace": lambda x: st.laplace.pdf(x, 0, np.sqrt(1/2)),
        "uniform": lambda x: st.uniform.pdf(x, -np.sqrt(3), 2*np.sqrt(3))
    }

    y = pdf_functions[name](x)

    plt.plot(x, y, color="red", label="Теория")
    plt.title(f"Распределение {name}, n={n}")
    plt.xlabel("Значение случайной величины")
    plt.ylabel("Плотность вероятности")
    plt.legend()
