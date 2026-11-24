#!/usr/bin/env python3
"""Gráficas de la tarea 4 (pasos e y g) a partir de build/results/results.csv.

Uso (desde knn2/):  python3 graficas.py [ruta_csv] [directorio_salida]
Por defecto lee build/results/results.csv y escribe en graficas/.
Solo necesita matplotlib (no pandas).

Si una configuración se ejecutó varias veces, se usa la ÚLTIMA fila del CSV
(las repeticiones se añaden al final).
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "build/results/results.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "graficas"
os.makedirs(OUT, exist_ok=True)

# Paleta validada (categórica, orden fijo) + rampa azul ordinal para los tamaños.
C_DATA, C_ASC, C_DESC = "#2a78d6", "#eb6834", "#1baf7a"
SIZE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]
GRAY, INK, MUTED = "#8a8a85", "#1f1f1e", "#5f5e5a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.edgecolor": "#b8b7b0", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})
NTRAIN = 60000  # tamaño de train en la evaluación final


# --------------------------------------------------------------------------- datos
def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("procs", "evalProcs", "nTune", "folds", "maxK", "nTest", "bestK", "wasted"):
                r[k] = int(float(r[k]))
            for k in ("accuracy", "total", "tune", "eval", "compute", "comm"):
                r[k] = float(r[k])
            key = (r["mode"], r["procs"], r["evalProcs"], r["nTune"], r["nTest"], r["order"])
            rows[key] = r  # la última repetición sustituye a las anteriores
    return list(rows.values())


def auto_eval_procs(r):
    """Mismo cálculo que autoEvalProcs() en knn_mpi.cc."""
    t = r["nTune"]
    tune = r["maxK"] * t * (t - t / r["folds"])
    ev = r["nTest"] * NTRAIN
    n = round(ev / (ev + tune) * r["procs"])
    return min(max(1, n), r["procs"] - 1)


rows = load(CSV)
sizes = sorted({(r["nTune"], r["nTest"]) for r in rows})
seq = {(r["nTune"], r["nTest"]): r for r in rows if r["mode"] == "seq"}


def size_label(s):
    return f"nTune={s[0]}, nTest={s[1]}"


def series(mode, size, order="asc"):
    """Filas de un modo y tamaño ordenadas por P (en spec, solo el reparto automático)."""
    out = [r for r in rows if r["mode"] == mode and (r["nTune"], r["nTest"]) == size
           and r["order"] == order]
    if mode == "spec":
        out = [r for r in out if r["evalProcs"] == auto_eval_procs(r)]
    return sorted(out, key=lambda r: r["procs"])


def speedup(r):
    return seq[(r["nTune"], r["nTest"])]["total"] / r["total"]


all_p = sorted({r["procs"] for r in rows if r["mode"] != "seq"})
pmax = max(all_p)


def ht_region(ax):
    """Sombrea P > 4: a partir de ahí son hilos hardware (hyperthreading), no núcleos."""
    ax.axvspan(4.15, pmax + 0.5, color="#f1f0ec", zorder=0, lw=0)
    y0, y1 = ax.get_ylim()
    ax.text(pmax + 0.4, y0 + 0.03 * (y1 - y0), "hyperthreading\n(P > 4 núcleos físicos)",
            ha="right", va="bottom", fontsize=8.5, color=MUTED)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=180)
    plt.close(fig)
    print("  ", name)


# --------------------------------------------------------------------------- tabla
with open(os.path.join(OUT, "tabla_resumen.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["modo", "orden", "P", "E_eval", "nTune", "nTest", "bestK", "accuracy",
                "T_total_s", "T_ajuste_s", "T_eval_s", "speedup", "eficiencia",
                "comunicacion_%", "ocioso_%", "muestras_descartadas"])
    for r in sorted(rows, key=lambda r: (r["nTune"], r["mode"], r["order"], r["procs"], r["evalProcs"])):
        s = speedup(r)
        busy = r["procs"] * r["total"]
        idle = max(0.0, busy - r["compute"] - r["comm"])
        w.writerow([r["mode"], r["order"], r["procs"], r["evalProcs"], r["nTune"], r["nTest"],
                    r["bestK"], r["accuracy"], f"{r['total']:.2f}", f"{r['tune']:.2f}",
                    f"{r['eval']:.2f}", f"{s:.2f}", f"{s / r['procs']:.2f}",
                    f"{100 * r['comm'] / busy:.1f}", f"{100 * idle / busy:.1f}", r["wasted"]])
print("Gráficas generadas en", OUT)
print("   tabla_resumen.csv")

# --------------------------------------------------------------------------- 1 y 2
for metric, name, ylabel, title in (
        ("speedup", "1_speedup_data.png", "Speed-up  (T secuencial / T paralelo)",
         "Speed-up del modo data según P y tamaño del problema"),
        ("eff", "2_eficiencia_data.png", "Eficiencia  (speed-up / P)",
         "Eficiencia del modo data según P y tamaño del problema")):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ps = [1] + all_p
    if metric == "speedup":
        ax.plot(ps, ps, ls="--", color=GRAY, lw=1.4, label="Ideal (lineal)")
    else:
        ax.axhline(1.0, ls="--", color=GRAY, lw=1.4, label="Ideal")
    for i, s in enumerate(sizes):
        rs = series("data", s)
        if not rs or s not in seq:
            continue
        x = [1] + [r["procs"] for r in rs]
        y = [1.0] + [speedup(r) if metric == "speedup" else speedup(r) / r["procs"] for r in rs]
        ax.plot(x, y, marker="os^"[i % 3], color=SIZE_RAMP[min(i, 2)], label=size_label(s))
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), textcoords="offset points", xytext=(6, -3),
                    fontsize=8.5, color=INK)
    ax.set_xticks(ps)
    ax.set_xlabel("Procesos MPI (P)")
    ax.set_ylabel(ylabel)
    if metric == "eff":
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(0, pmax + 0.5)
    ax.set_xlim(0.7, pmax + 0.5)
    ht_region(ax)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left" if metric == "speedup" else "lower left")
    finish(fig, name)

# --------------------------------------------------------------------------- 3
fig, axes = plt.subplots(1, len(sizes), figsize=(4.3 * len(sizes), 4.4), sharey=True)
axes = axes if len(sizes) > 1 else [axes]
for ax, s in zip(axes, sizes):
    ps = [1] + all_p
    ax.plot(ps, ps, ls="--", color=GRAY, lw=1.2, label="Ideal")
    for mode, order, color, mk, lab in (("data", "asc", C_DATA, "o", "data"),
                                        ("spec", "asc", C_ASC, "s", "spec (K ascendente)"),
                                        ("spec", "desc", C_DESC, "^", "spec (K descendente)")):
        rs = series(mode, s, order)
        if rs and s in seq:
            ax.plot([1] + [r["procs"] for r in rs], [1.0] + [speedup(r) for r in rs],
                    marker=mk, color=color, label=lab)
    ax.set_title(size_label(s), loc="left", fontsize=10.5)
    ax.set_xticks(ps)
    ax.set_xlabel("Procesos MPI (P)")
    ax.set_xlim(0.7, pmax + 0.5)
    ax.set_ylim(0, pmax + 0.5)
axes[0].set_ylabel("Speed-up")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle("Speed-up por estrategia: paralelismo de datos frente a especulativo",
             x=0.01, ha="left", fontweight="bold", fontsize=11.5)
finish(fig, "3_speedup_por_estrategia.png")

# --------------------------------------------------------------------------- 4
# Reparto del tiempo de CPU disponible (P · T_total) en el tamaño intermedio.
mid = sizes[len(sizes) // 2] if len(sizes) > 1 else sizes[0]
per_sample = seq[mid]["eval"] / mid[1]  # coste aprox. de clasificar una muestra de test
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
parts = [("Cómputo útil", C_DATA), ("Cómputo descartado (especulación)", C_ASC),
         ("Comunicación (dentro de MPI)", "#eda100"), ("Ocioso / espera", "#c9c8c1")]
for ax, (mode, order, title) in zip(axes, (("data", "asc", "Modo data"),
                                           ("spec", "asc", "Modo spec (K ascendente)"))):
    rs = series(mode, mid, order)
    xs = [str(r["procs"]) for r in rs]
    bottoms = [0.0] * len(rs)
    for idx, (lab, col) in enumerate(parts):
        vals = []
        for r in rs:
            busy = r["procs"] * r["total"]
            wasted = min(r["wasted"] * per_sample, r["compute"])
            useful = r["compute"] - wasted
            idle = max(0.0, busy - r["compute"] - r["comm"])
            vals.append(100 * (useful, wasted, r["comm"], idle)[idx] / busy)
        ax.bar(xs, vals, bottom=bottoms, color=col, width=0.6, edgecolor="white", linewidth=2,
               label=lab)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xlabel("Procesos MPI (P)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("% del tiempo de CPU disponible (P · T_total)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"Reparto del tiempo: cómputo, comunicación y espera ({size_label(mid)})",
             x=0.01, ha="left", fontweight="bold", fontsize=11.5)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(OUT, "4_reparto_tiempo.png"), dpi=180)
plt.close(fig)
print("   4_reparto_tiempo.png")

# --------------------------------------------------------------------------- 5
sweep = sorted([r for r in rows if r["mode"] == "spec" and r["order"] == "asc"
                and r["procs"] == pmax and (r["nTune"], r["nTest"]) == mid],
               key=lambda r: r["evalProcs"])
if len(sweep) > 1:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    auto = auto_eval_procs(sweep[0])
    cols = [C_ASC if r["evalProcs"] == auto else "#f2b08f" for r in sweep]
    bars = ax.bar([str(r["evalProcs"]) for r in sweep], [r["total"] for r in sweep], color=cols,
                  width=0.6)
    for b, r in zip(bars, sweep):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                f"{r['total']:.1f} s\n{r['wasted']} descartadas", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ref = [r for r in series("data", mid) if r["procs"] == pmax]
    if ref:
        ax.axhline(ref[0]["total"], color=C_DATA, ls="--", lw=1.6)
        ax.text(-0.45, ref[0]["total"] + 0.6, f"data con P={pmax}: {ref[0]['total']:.1f} s",
                ha="left", va="bottom", color=C_DATA, fontsize=9, fontweight="bold")
    ax.set_xlabel(f"Procesos de evaluación E (de P = {pmax});  color intenso = reparto automático")
    ax.set_ylabel("Tiempo total (s)")
    ax.set_ylim(0, max(r["total"] for r in sweep) * 1.25)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Modo spec: reparto entre ajuste y evaluación ({size_label(mid)})", loc="left")
    finish(fig, "5_barrido_procesos_evaluacion.png")

# --------------------------------------------------------------------------- 6
fig, ax = plt.subplots(figsize=(7.2, 4.4))
xl = [f"{s[0]}/{s[1]}" for s in sizes]
for mode, color, mk in (("data", C_DATA, "o"), ("spec", C_ASC, "s")):
    for p, ls in ((4, ":"), (pmax, "-")):
        pts = []
        for i, s in enumerate(sizes):
            r = [x for x in series(mode, s) if x["procs"] == p]
            if r and s in seq:
                pts.append((i, speedup(r[0])))
        if len(pts) > 1:
            ax.plot([q[0] for q in pts], [q[1] for q in pts], color=color, ls=ls, marker=mk,
                    label=f"{mode}{' (K asc.)' if mode == 'spec' else ''}, P = {p}")
ax.set_xticks(range(len(sizes)))
ax.set_xticklabels(xl)
ax.set_xlabel("Tamaño del problema (nTune / nTest)")
ax.set_ylabel("Speed-up")
ax.set_ylim(0, None)
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Speed-up según el tamaño del problema", loc="left")
finish(fig, "6_speedup_vs_tamano.png")
