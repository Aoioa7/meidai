import numpy as np
import matplotlib.pyplot as plt

def plot_numerical_vs_exact():
    x = np.linspace(0, 1, 11)

    # Cコードの出力をベタ打ち
    u_num = {
        0.00: [0.0000, 0.3090, 0.5878, 0.8090, 0.9511, 1.0000, 0.9511, 0.8090, 0.5878, 0.3090, 0.0000],
        0.05: [0.0000, 0.1871, 0.3559, 0.4898, 0.5758, 0.6054, 0.5758, 0.4898, 0.3559, 0.1871, 0.0000],
        0.10: [0.0000, 0.1133, 0.2154, 0.2965, 0.3486, 0.3665, 0.3486, 0.2965, 0.2154, 0.1133, 0.0000],
        0.15: [0.0000, 0.0686, 0.1304, 0.1795, 0.2111, 0.2219, 0.2111, 0.1795, 0.1304, 0.0686, 0.0000],
        0.20: [0.0000, 0.0415, 0.0790, 0.1087, 0.1278, 0.1344, 0.1278, 0.1087, 0.0790, 0.0415, 0.0000],
        0.25: [0.0000, 0.0251, 0.0478, 0.0658, 0.0774, 0.0813, 0.0774, 0.0658, 0.0478, 0.0251, 0.0000],
        0.30: [0.0000, 0.0152, 0.0289, 0.0398, 0.0468, 0.0492, 0.0468, 0.0398, 0.0289, 0.0152, 0.0000],
    }

    plt.figure(figsize=(8, 5))
    for t, u in u_num.items():
        # 数値解（実線＋マーカー）
        plt.plot(x, u, marker='o', linestyle='-', label=f"num t={t:.2f}s")
        # 厳密解 u_e(x,t) = exp(-π² t) sin(π x)
        u_exact = np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
        plt.plot(x, u_exact, linestyle='--', label=f"exact t={t:.2f}s")

    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("u(x) by t: Numerical vs Exact")
    plt.legend(ncol=2, fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_numerical_vs_exact()
