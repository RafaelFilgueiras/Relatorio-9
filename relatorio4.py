# -- coding: utf-8 --
"""
Comparação entre RK2, Taylor-4 e RK4 para o PVI:
    y' = -y + x + 2,  y(0) = 2
Solução exata: y(x) = exp(-x) + x + 1

Saídas:
1) Tabela com (método, h, y(0.3), erro absoluto, erro relativo) + CSV
2) Gráfico comparando y_exato(x) e as curvas (com h = 0.1) + PNG
"""

from __future__ import annotations
import math
from typing import Callable, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------- Definições do problema ----------------------
def f(x: float, y: float) -> float:
    """Campo do PVI: y' = -y + x + 2."""
    return -y + x + 2

def y_exata(x: float) -> float:
    """Solução exata: y(x) = e^{-x} + x + 1."""
    return math.exp(-x) + x + 1


# ---------------------- Métodos Numéricos ---------------------------
def rk2(f: Callable[[float, float], float],
        x0: float, y0: float, h: float, xf: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runge–Kutta de 2ª ordem (Euler melhorado/Heun).
    Retorna arrays (xs, ys) com N+1 pontos de [x0, xf].
    """
    N = int(round((xf - x0)/h))
    xs = np.zeros(N+1, dtype=float)
    ys = np.zeros(N+1, dtype=float)
    xs[0], ys[0] = x0, y0
    for n in range(N):
        x, y = xs[n], ys[n]
        k1 = h * f(x, y)
        k2 = h * f(x + h, y + k1)
        ys[n+1] = y + 0.5*(k1 + k2)
        xs[n+1] = x + h
    return xs, ys


def rk4(f: Callable[[float, float], float],
        x0: float, y0: float, h: float, xf: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runge–Kutta de 4ª ordem clássico.
    Retorna arrays (xs, ys) com N+1 pontos de [x0, xf].
    """
    N = int(round((xf - x0)/h))
    xs = np.zeros(N+1, dtype=float)
    ys = np.zeros(N+1, dtype=float)
    xs[0], ys[0] = x0, y0
    for n in range(N):
        x, y = xs[n], ys[n]
        k1 = h * f(x, y)
        k2 = h * f(x + 0.5*h, y + 0.5*k1)
        k3 = h * f(x + 0.5*h, y + 0.5*k2)
        k4 = h * f(x + h, y + k3)
        ys[n+1] = y + (k1 + 2*k2 + 2*k3 + k4)/6.0
        xs[n+1] = x + h
    return xs, ys


def taylor4_linear_problem(x0: float, y0: float, h: float, xf: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Método de Taylor de 4ª ordem especializado para o PVI:
        y' = -y + x + 2
    Derivadas simbólicas (para este PVI):
        y'    = -y + x + 2
        y''   =  y - x - 1
        y'''  = -y + x + 1
        y'''' =  y - x - 1
    Retorna arrays (xs, ys).
    """
    N = int(round((xf - x0)/h))
    xs = np.zeros(N+1, dtype=float)
    ys = np.zeros(N+1, dtype=float)
    xs[0], ys[0] = x0, y0

    for n in range(N):
        x, y = xs[n], ys[n]
        y1 = -y + x + 2       # y'
        y2 =  y - x - 1       # y''
        y3 = -y + x + 1       # y'''
        y4 =  y - x - 1       # y''''
        ys[n+1] = y + h*y1 + (h*2/2.0)*y2 + (h*3/6.0)*y3 + (h*4/24.0)*y4
        xs[n+1] = x + h

    return xs, ys


# ---------------------- Rotinas auxiliares --------------------------
def erro_abs_rel(y_ex: float, y_ap: float) -> Tuple[float, float]:
    eabs = abs(y_ex - y_ap)
    erel = eabs / max(1.0, abs(y_ex))
    return eabs, erel


def experimento(h_vals: List[float],
                x0: float, y0: float, xf: float) -> pd.DataFrame:
    registros = []
    yext = y_exata(xf)
    for h in h_vals:
        # RK2
        xs2, ys2 = rk2(f, x0, y0, h, xf)
        y2f = ys2[-1]
        eabs, erel = erro_abs_rel(yext, y2f)
        registros.append(("RK2", h, y2f, eabs, erel))

        # Taylor-4 (especializado)
        xst, yst = taylor4_linear_problem(x0, y0, h, xf)
        ytf = yst[-1]
        eabs, erel = erro_abs_rel(yext, ytf)
        registros.append(("Taylor-4", h, ytf, eabs, erel))

        # RK4
        xs4, ys4 = rk4(f, x0, y0, h, xf)
        y4f = ys4[-1]
        eabs, erel = erro_abs_rel(yext, y4f)
        registros.append(("RK4", h, y4f, eabs, erel))

    df = pd.DataFrame(registros,
                      columns=["Método", "h", "y(0.3) aprox", "Erro absoluto", "Erro relativo"])
    return df.sort_values(by=["Método", "h"]).reset_index(drop=True)


def plot_comparacao(x0: float, y0: float, xf: float, h_plot: float,
                    fig_path: str | None = "comparacao_metodos_RK2_T4_RK4.png") -> None:
    """Plota solução exata e as curvas de RK2/Taylor-4/RK4 com h=h_plot."""
    xs2, ys2 = rk2(f, x0, y0, h_plot, xf)
    xst, yst = taylor4_linear_problem(x0, y0, h_plot, xf)
    xs4, ys4 = rk4(f, x0, y0, h_plot, xf)

    xs_dense = np.linspace(x0, xf, 400)
    ys_dense = np.array([y_exata(x) for x in xs_dense])

    plt.figure(figsize=(8, 5))
    plt.plot(xs_dense, ys_dense, label="Solução exata")
    plt.plot(xs2, ys2, "o--", label=f"RK2 (h={h_plot})")
    plt.plot(xst, yst, "s--", label=f"Taylor-4 (h={h_plot})")
    plt.plot(xs4, ys4, "^--", label=f"RK4 (h={h_plot})")
    plt.xlabel("x"); plt.ylabel("y(x)")
    plt.title("Comparação: solução exata vs. RK2 / Taylor-4 / RK4")
    plt.grid(True); plt.legend(); plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=150)
    plt.show()


# ---------------------- Execução principal --------------------------
if __name__ == "__main__":
    # Parâmetros do PVI
    x0, y0 = 0.0, 2.0
    xf = 0.3
    h_vals = [0.1, 0.05, 0.025]

    # Experimento e tabela
    df = experimento(h_vals, x0, y0, xf)
    # Exibe tabela no terminal
    with pd.option_context("display.float_format", lambda v: f"{v:.12g}"):
        print("\nTabela: método, h, y(0.3), erro absoluto, erro relativo\n")
        print(df.to_string(index=False))

    # Salva CSV
    csv_path = "Resultados_RK2_T4_RK4.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nArquivo salvo: {csv_path}")

    # Gráfico (h = 0.1)
    plot_comparacao(x0, y0, xf, h_plot=0.1,
                    fig_path="comparacao_metodos_RK2_T4_RK4.png")
    print("Figura salva: comparacao_metodos_RK2_T4_RK4.png")
