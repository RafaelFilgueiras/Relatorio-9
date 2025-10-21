# -*- coding: utf-8 -*-
"""
Método de Newton multivariável para sistemas não lineares (2x2)
- Sistema de teste do relatório:
    f1(x,y) = x^2 + y^2 - 4
    f2(x,y) = exp(x) + y - 1
  Raiz ≈ interseção entre x^2 + y^2 = 4 e y = 1 - exp(x)

Funcionalidades:
- Itera Newton até tolerância (resíduo e passo)
- Exporta CSV com (k, xk, yk, ||F||, ||Δx||)
- Gera gráfico (PNG e PDF) com as curvas e o ponto encontrado
- Estrutura clara para trocar o sistema (basta alterar f1, f2 e Jacobiano)

Requisitos:
- numpy, matplotlib, pandas (opcional: se pandas não estiver disponível, o CSV é salvo via numpy.savetxt)

"""

from __future__ import annotations
import numpy as np
import math
import os

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

import matplotlib.pyplot as plt


# =========================
# 1) Definição do sistema
# =========================
def F(xy: np.ndarray) -> np.ndarray:
    """Vetor de funções F(x,y) = [f1, f2]."""
    x, y = xy
    f1 = x**2 + y**2 - 4.0
    f2 = math.exp(x) + y - 1.0
    return np.array([f1, f2], dtype=float)


def J(xy: np.ndarray) -> np.ndarray:
    """Jacobiana J(x,y)."""
    x, y = xy
    return np.array([[2.0*x, 2.0*y],
                     [math.exp(x), 1.0]], dtype=float)


# Opcional: Jacobiana numérica por diferenças finitas (útil se trocarem o sistema e não quiserem derivar à mão)
def J_numeric(xy: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Jacobiana numérica por diferenças centrais."""
    n = xy.size
    m = F(xy).size
    Jmat = np.zeros((m, n), dtype=float)
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        fp = F(xy + h*e)
        fm = F(xy - h*e)
        Jmat[:, j] = (fp - fm) / (2.0*h)
    return Jmat


# =======================================
# 2) Método de Newton (genérico 2 variáveis)
# =======================================
def newton_system(
    x0: np.ndarray,
    tol_res: float = 1e-10,
    tol_step: float = 1e-10,
    maxit: int = 50,
    use_numeric_jacobian: bool = False,
    damping: bool = False
):
    """
    Resolve F(x)=0 via Newton:
        J(xk) Δx = -F(xk); x_{k+1} = xk + Δx

    Parâmetros:
        x0: chute inicial (array shape (2,))
        tol_res: tolerância para norma do resíduo ||F||
        tol_step: tolerância para norma do passo ||Δx||
        maxit: máximo de iterações
        use_numeric_jacobian: se True, usa J numérica; se False, J analítica
        damping: se True, faz amortecimento simples na atualização

    Retorno:
        x: solução aproximada
        hist: lista de dicionários com histórico por iteração
        convergiu: bool
        k: iterações usadas
    """
    x = np.array(x0, dtype=float)
    hist = []

    for k in range(maxit+1):
        Fx = F(x)
        res_norm = np.linalg.norm(Fx, ord=2)

        if k == 0:
            step_norm = np.nan
        if res_norm < tol_res:
            hist.append({"k": k, "x": x[0], "y": x[1], "res_norm": res_norm, "step_norm": 0.0})
            return x, hist, True, k

        # Jacobiana
        Jx = J_numeric(x) if use_numeric_jacobian else J(x)

        # Resolve J Δx = -F
        try:
            dx = np.linalg.solve(Jx, -Fx)
        except np.linalg.LinAlgError:
            # Jacobiana singular/não-invertível
            hist.append({"k": k, "x": x[0], "y": x[1], "res_norm": res_norm, "step_norm": np.nan})
            return x, hist, False, k

        step_norm = np.linalg.norm(dx, ord=2)

        # Amortecimento opcional (backtracking simples)
        if damping:
            alpha = 1.0
            x_trial = x + alpha*dx
            # Reduz alpha enquanto não melhora suficientemente ||F||
            while np.linalg.norm(F(x_trial), 2) > (1 - 1e-4*alpha) * res_norm and alpha > 1e-6:
                alpha *= 0.5
                x_trial = x + alpha*dx
            x_new = x_trial
        else:
            x_new = x + dx

        hist.append({"k": k, "x": x[0], "y": x[1], "res_norm": res_norm, "step_norm": step_norm})
        x = x_new

        if step_norm < tol_step:
            # registra último estado
            Fx = F(x)
            res_norm = np.linalg.norm(Fx, ord=2)
            hist.append({"k": k+1, "x": x[0], "y": x[1], "res_norm": res_norm, "step_norm": 0.0})
            return x, hist, True, k+1

    # Se chegar aqui, não convergiu no limite de iterações
    return x, hist, False, maxit


# =======================================
# 3) Utilitários: salvar CSV
# =======================================
def save_history_csv(hist, path_csv: str = "newton_sistema_historico.csv"):
    """Salva histórico de iterações em CSV."""
    rows = []
    for row in hist:
        rows.append([row["k"], row["x"], row["y"], row["res_norm"], row["step_norm"]])

    if HAS_PANDAS:
        df = pd.DataFrame(rows, columns=["k", "xk", "yk", "||F||2", "||Δx||2"])
        df.to_csv(path_csv, index=False, float_format="%.16e")
    else:
        header = "k,xk,yk,||F||2,||Δx||2"
        np.savetxt(path_csv, np.array(rows, dtype=float), delimiter=",", header=header, comments="", fmt="%.16e")

    return os.path.abspath(path_csv)


# =======================================
# 4) Plot das curvas e do ponto de interseção
# =======================================
def plot_system_solution(x_sol: np.ndarray,
                         x_range=(-3.0, 3.0),
                         y_range=(-3.0, 3.0),
                         nx=400,
                         ny=400,
                         fname_png="sistema_nao_linear.png",
                         fname_pdf="sistema_nao_linear.pdf"):
    """
    Plota:
      - Circunferência x^2 + y^2 = 4 (nível f1=0)
      - Curva y = 1 - exp(x) (nível f2=0)
      - Ponto de solução
    """
    # Malha para contornos
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    X, Y = np.meshgrid(xs, ys)

    F1 = X**2 + Y**2 - 4.0
    F2 = np.exp(X) + Y - 1.0

    plt.figure(figsize=(7, 6))
    # Contorno nível zero das duas funções
    c1 = plt.contour(X, Y, F1, levels=[0.0])
    c2 = plt.contour(X, Y, F2, levels=[0.0])

    # Ponto da solução
    plt.plot([x_sol[0]], [x_sol[1]], 'o', label="Solução (Newton)", markersize=7)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sistemas Não Lineares: f1=0 e f2=0")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(fname_png, dpi=200)
    plt.savefig(fname_pdf)
    plt.close()

    return os.path.abspath(fname_png), os.path.abspath(fname_pdf)


# =======================================
# 5) Execução padrão
# =======================================
if __name__ == "__main__":
    # Chute inicial (pode ser ajustado conforme análise gráfica)
    x0 = np.array([1.0, 0.0])

    # Rodar Newton
    x_sol, hist, ok, k = newton_system(
        x0,
        tol_res=1e-12,
        tol_step=1e-12,
        maxit=50,
        use_numeric_jacobian=False,  # mude para True se trocarem o sistema e não tiverem J analítica
        damping=False                 # mude para True se houver dificuldade de convergência
    )

    # Relato no console
    if ok:
        print(f"[OK] Convergiu em {k} iterações.")
    else:
        print(f"[ATENÇÃO] Não convergiu em {k} iterações.")
    print(f"Solução aproximada: x = {x_sol[0]:.12f}, y = {x_sol[1]:.12f}")
    print(f"||F(x)||_2 = {np.linalg.norm(F(x_sol),2):.3e}")

    # Salvar histórico
    csv_path = save_history_csv(hist, "newton_sistema_historico.csv")
    print(f"CSV salvo em: {csv_path}")

    # Plot
    png_path, pdf_path = plot_system_solution(x_sol, fname_png="sistema_nao_linear.png",
                                              fname_pdf="sistema_nao_linear.pdf")
    print(f"Figura PNG: {png_path}")
    print(f"Figura PDF: {pdf_path}")

    # Observação para o relatório:
    # A tabela de iterações está no CSV gerado.
    # As figuras (PNG e PDF) mostram as curvas f1=0 (circunferência) e f2=0 (exponencial deslocada),
    # com o ponto de interseção marcado (solução de F(x)=0).
