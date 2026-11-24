"""Grafos de la solución paralela MPI (tarea 4, apartado f).

Genera grafo_mpi_data.(png|svg) y grafo_mpi_spec.(png|svg) en el directorio indicado:
    python3 diagramas_mpi.py [directorio_salida]
"""
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams["font.family"] = "DejaVu Sans"

# Paleta
C_COMP = "#dbe8f6"   # cómputo
C_COMP_E = "#3b6ea5"
C_COLL = "#f6b26b"   # colectiva / sincronización
C_COLL_E = "#b45f06"
C_P2P = "#38761d"    # punto a punto
C_SPEC = "#fff2cc"   # trabajo especulativo
C_SPEC_E = "#bf9000"
C_INV = "#f4cccc"    # invalidación / reinicio
C_INV_E = "#990000"
C_IDLE = "#eeeeee"   # espera / ocioso
C_HEAD = "#1f3b57"
C_DEP = "#cc0000"
FS = 8.3


class Diagram:
    def __init__(self, width, height, title, box_w=2.6):
        self.bw = box_w
        self.fig, self.ax = plt.subplots(figsize=(width * 0.74, height * 0.74))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=6)
        self.W, self.H = width, height

    def lifelines(self, xs, y0, y1):
        for x in xs:
            self.ax.plot([x, x], [y0, y1], color="#9a9a9a", lw=1, ls=(0, (2, 3)), zorder=0)

    def header(self, x, y, text, w=2.6, h=0.7):
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                         boxstyle="round,pad=0.02,rounding_size=0.12",
                                         fc=C_HEAD, ec=C_HEAD, zorder=3))
        self.ax.text(x, y, text, ha="center", va="center", color="white",
                     fontsize=FS, fontweight="bold", zorder=4, linespacing=1.15)

    def box(self, x, y, h, text, fc=C_COMP, ec=C_COMP_E, w=None, hatch=None, ls="-",
            bold=False, fs=FS):
        w = w or self.bw
        self.ax.add_patch(FancyBboxPatch((x - w / 2, y), w, h,
                                         boxstyle="round,pad=0.02,rounding_size=0.1",
                                         fc=fc, ec=ec, lw=1.1, hatch=hatch, ls=ls, zorder=2))
        # Fondo liso bajo el texto si la caja va rayada, para que se lea.
        bb = dict(fc=fc, ec="none", pad=1.2) if hatch else None
        self.ax.text(x, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4,
                     fontweight="bold" if bold else "normal", linespacing=1.2, bbox=bb)

    def bar(self, x0, x1, y, h, text, fs=FS):
        self.ax.add_patch(Rectangle((x0, y), x1 - x0, h, fc=C_COLL, ec=C_COLL_E, lw=1.2,
                                    zorder=2))
        self.ax.text((x0 + x1) / 2, y + h / 2, text, ha="center", va="center",
                     fontsize=fs, fontweight="bold", zorder=4)

    def frame(self, x0, y0, x1, y1, label):
        self.ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#555555",
                                    lw=1.1, ls=(0, (5, 3)), zorder=1))
        tw = 0.125 * len(label) + 0.4
        self.ax.add_patch(Rectangle((x0, y0), tw, 0.36, fc="#555555", ec="#555555", zorder=1.5))
        self.ax.text(x0 + 0.12, y0 + 0.18, label, ha="left", va="center", fontsize=FS - 0.5,
                     color="white", fontweight="bold", zorder=4)

    def arrow(self, x0, y0, x1, y1, text=None, color=C_P2P, text_dy=-0.16, ls="-", fs=FS):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=5,
                         arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, ls=ls,
                                         shrinkA=0, shrinkB=0, mutation_scale=13))
        if text:
            self.ax.text((x0 + x1) / 2, (y0 + y1) / 2 + text_dy, text, ha="center",
                         va="bottom", fontsize=fs, color=color, fontweight="bold", zorder=6,
                         bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))

    def phase(self, x, y0, y1, text):
        self.ax.plot([x + 0.18, x, x, x + 0.18], [y0, y0, y1, y1], color="#444", lw=1.2)
        self.ax.text(x - 0.12, (y0 + y1) / 2, text, rotation=90, ha="right", va="center",
                     fontsize=FS + 0.3, fontweight="bold", color="#444")

    def dots(self, x, y):
        self.ax.text(x, y, "···", ha="center", va="center", fontsize=14, color="#555",
                     fontweight="bold")

    def legend(self, items, y):
        handles = []
        for kind, label in items:
            if kind == "comp":
                handles.append(Patch(fc=C_COMP, ec=C_COMP_E, label=label))
            elif kind == "coll":
                handles.append(Patch(fc=C_COLL, ec=C_COLL_E, label=label))
            elif kind == "spec":
                handles.append(Patch(fc=C_SPEC, ec=C_SPEC_E, hatch="///", label=label))
            elif kind == "inv":
                handles.append(Patch(fc=C_INV, ec=C_INV_E, label=label))
            elif kind == "idle":
                handles.append(Patch(fc=C_IDLE, ec="#888", ls="--", label=label))
            elif kind == "p2p":
                handles.append(Line2D([0], [0], color=C_P2P, lw=2, label=label))
            elif kind == "dep":
                handles.append(Line2D([0], [0], color=C_DEP, lw=2, ls="--", label=label))
        self.ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - y / self.H),
                       bbox_transform=self.ax.transAxes, ncol=4, fontsize=FS, frameon=False,
                       handlelength=2.2, columnspacing=1.6)

    def save(self, name):
        self.fig.tight_layout()
        for ext in ("png", "svg"):
            self.fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Modo data: paralelismo de datos (SPMD)
# ---------------------------------------------------------------------------
def diagram_data():
    d = Diagram(12.0, 15.0, "Solución paralela MPI · modo data (paralelismo de datos, SPMD)")
    xs = [2.6, 6.2, 9.8]
    d.lifelines(xs, 1.2, 13.6)
    d.header(xs[0], 0.75, "P0\n(raíz)")
    d.header(xs[1], 0.75, "Pr\n(r = 1 … P−2)")
    d.header(xs[2], 0.75, "P(P−1)")
    d.dots((xs[0] + xs[1]) / 2, 0.75)
    d.dots((xs[1] + xs[2]) / 2, 0.75)
    L, R = xs[0] - 1.45, xs[2] + 1.45

    d.bar(L, R, 1.45, 0.5, "MPI_Init · MPI_Barrier  →  t0 = MPI_Wtime()   (sincronización inicial)")
    for x in xs:
        d.box(x, 2.2, 0.9, "Carga MNIST\n(60 000 train + nTest)\nreplicada en cada proceso")
    d.phase(0.75, 1.45, 3.1, "Inicio")

    # Fase 1
    d.frame(L - 0.15, 3.45, R + 0.15, 8.3, "loop  K = 1 … maxK   ×   fold = 0 … folds−1")
    for x in xs:
        d.box(x, 4.0, 0.8, "splitFold + KNN(train fold)\n(replicado, sin comunicar)")
    d.box(xs[0], 5.0, 0.95, "Clasifica validación\n[0 , n/P)")
    d.box(xs[1], 5.0, 0.95, "Clasifica validación\n[n·r/P , n·(r+1)/P)")
    d.box(xs[2], 5.0, 0.95, "Clasifica validación\n[n·(P−1)/P , n)")
    d.bar(L, R, 6.2, 0.55, "MPI_Allreduce(SUM, aciertos)   ·   sincronización por fold   (maxK·folds veces)")
    for x in xs:
        d.box(x, 7.0, 0.95, "acc(K) idéntica en todos\n→ actualiza bestK\n(local, sin comunicar)")
    d.phase(0.75, 3.45, 8.3, "Fase 1 · ajuste de K")

    # Dependencia
    d.ax.plot([L, R], [8.95, 8.95], color=C_DEP, lw=1.8, ls="--", zorder=3)
    d.ax.text((L + R) / 2, 8.85, "Dependencia de datos: la evaluación necesita el bestK definitivo",
              ha="center", va="bottom", fontsize=FS, color=C_DEP, fontweight="bold",
              bbox=dict(fc="white", ec="none", pad=0.5))

    # Fase 2
    d.box(xs[0], 9.45, 1.0, "KNN(train, bestK)\nClasifica test [0 , m/P)")
    d.box(xs[1], 9.45, 1.0, "KNN(train, bestK)\nClasifica test\n[m·r/P , m·(r+1)/P)")
    d.box(xs[2], 9.45, 1.0, "KNN(train, bestK)\nClasifica test\n[m·(P−1)/P , m)")
    d.bar(L, R, 10.75, 0.5, "MPI_Reduce(SUM, aciertos) → P0")
    d.phase(0.75, 9.45, 11.25, "Fase 2 · evaluación")

    # Cierre
    d.bar(L, R, 11.65, 0.5, "MPI_Reduce(MAX, t_fin) → P0   ·   MPI_Gather(estadísticas) → P0")
    d.box(xs[0], 12.4, 0.6, "Imprime resultados\n(línea RESULT)")
    d.bar(L, R, 13.25, 0.45, "MPI_Finalize")
    d.phase(0.75, 11.65, 13.7, "Recogida")

    d.ax.text(L, 14.05, "n = tamaño del fold de validación (nTune/folds)   ·   m = nTest   ·   "
              "reparto estático por bloques contiguos", fontsize=FS, color="#444")
    d.legend([("comp", "Cómputo local"), ("coll", "Comunicación colectiva / sincronización"),
              ("dep", "Dependencia entre fases")], 14.35)
    d.save("grafo_mpi_data")


# ---------------------------------------------------------------------------
# Modo spec: paralelismo funcional + datos + ejecución especulativa
# ---------------------------------------------------------------------------
def diagram_spec():
    d = Diagram(16.6, 17.0, "Solución paralela MPI · modo spec (paralelismo funcional + "
                "ejecución especulativa)", box_w=3.15)
    xs = [2.75, 6.4, 10.6, 14.25]
    tL, tR = 0.95, 8.2      # grupo de ajuste
    eL, eR = 8.75, 16.05    # grupo de evaluación
    d.ax.add_patch(Rectangle((tL, 3.3), tR - tL, 9.95, fc="#f3f7fb", ec="none", zorder=-1))
    d.ax.add_patch(Rectangle((eL, 3.3), eR - eL, 9.95, fc="#fdf8ec", ec="none", zorder=-1))
    d.ax.text((tL + tR) / 2, 3.38, "Grupo de ajuste  (T = P − E procesos)", ha="center",
              va="top", fontsize=FS + 0.5, fontweight="bold", color=C_COMP_E)
    d.ax.text((eL + eR) / 2, 3.38, "Grupo de evaluación  (E procesos)",
              ha="center", va="top", fontsize=FS + 0.5, fontweight="bold", color=C_SPEC_E)

    d.lifelines(xs, 1.2, 15.6)
    d.header(xs[0], 0.75, "P0\n(raíz global\ny del ajuste)", w=3.15)
    d.header(xs[1], 0.75, "P(T−1)", w=3.15)
    d.header(xs[2], 0.75, "P(T)\n(raíz de la\nevaluación)", w=3.15)
    d.header(xs[3], 0.75, "P(P−1)", w=3.15)
    d.dots((xs[0] + xs[1]) / 2, 0.75)
    d.dots((xs[2] + xs[3]) / 2, 0.75)
    L, R = tL, eR

    d.bar(L, R, 1.45, 0.45, "MPI_Init · MPI_Barrier  →  t0 = MPI_Wtime()")
    for x in xs:
        d.box(x, 2.05, 0.65, "Carga MNIST (replicada)")
    d.bar(L, R, 2.82, 0.42, "MPI_Comm_split(color = ajuste | evaluación)  →  comunicador 'group'")

    # ---------------- Ajuste ----------------
    d.frame(tL + 0.1, 3.75, tR - 0.1, 8.85, "loop  K en orden asc | desc")
    for x in xs[:2]:
        d.box(x, 4.3, 1.0, "CV de K repartida en el grupo:\nsplitFold + KNN y clasifica\nsu bloque de validación")
    d.bar(tL + 0.2, tR - 0.2, 5.5, 0.5, "MPI_Allreduce(SUM) en 'group'  (× folds)", fs=FS - 0.2)
    for x in xs[:2]:
        d.box(x, 6.2, 0.75, "¿acc(K) mejora?\nsí → bestK = K")
    d.box(xs[0], 7.2, 1.0, "si mejora:\nMPI_Send NEWK(K)\na cada proceso de eval.", fc="#d9ead3",
          ec=C_P2P)
    d.box(xs[0], 9.25, 0.75, "MPI_Send FIN(bestK)\na cada proceso de eval.", fc="#d9ead3", ec=C_P2P)
    d.box(xs[1], 10.4, 1.4, "Termina su parte\n(ocioso hasta\nel cierre)", fc=C_IDLE, ec="#888",
          ls="--")
    d.box(xs[0], 10.4, 1.4, "Espera el resultado\nde la evaluación", fc=C_IDLE, ec="#888", ls="--")
    d.box(xs[0], 12.1, 0.7, "MPI_Recv(resultado)", fc="#d9ead3", ec=C_P2P)

    # ---------------- Evaluación ----------------
    for x in xs[2:]:
        d.box(x, 3.95, 0.85, "K = primer K del orden\nKNN(train completo, K)", hatch="///",
              fc=C_SPEC, ec=C_SPEC_E)
    d.frame(eL + 0.1, 5.05, eR - 0.1, 10.85, "loop  hasta FIN y bloque completo")
    for x in xs[2:]:
        d.box(x, 5.6, 0.8, "MPI_Iprobe + MPI_Recv\n(no bloqueantes,\ncada 'chunk' muestras)",
              fc="#d9ead3", ec=C_P2P, fs=FS - 0.3)
        d.box(x, 6.6, 1.05, "¿llegó K' ≠ K?\nsí → descarta su bloque\ny reinicia con K'",
              fc=C_INV, ec=C_INV_E)
        d.box(x, 7.85, 0.95, "Clasifica 'chunk' muestras\nde su bloque de test\n(especulativo)",
              hatch="///", fc=C_SPEC, ec=C_SPEC_E)
        d.box(x, 9.0, 1.65, "Bloque completo:\n¿FIN recibido?\nsí → sale del bucle\nno → MPI_Recv bloqueante\n(espera NEWK / FIN)",
              fs=FS - 0.3)
    d.bar(eL + 0.2, eR - 0.2, 11.2, 0.5, "MPI_Reduce(SUM, aciertos) en 'group' → P(T)", fs=FS - 0.2)
    d.box(xs[2], 12.0, 0.9, "MPI_Send(precisión,\nt_eval) → P0", fc="#d9ead3", ec=C_P2P)

    # ---------------- Mensajes punto a punto ----------------
    d.arrow(xs[0] + 1.6, 7.75, eL + 0.1, 8.2, "NEWK(K)  ·  asíncrono", text_dy=-0.24)
    d.ax.text((xs[0] + 1.6 + eL) / 2, 8.15, "se atiende en el\nsiguiente MPI_Iprobe",
              ha="center", va="top", fontsize=FS - 0.6, color=C_P2P, style="italic")
    d.arrow(xs[0] + 1.6, 9.62, eL + 0.1, 9.95, "FIN(bestK definitivo)", text_dy=-0.24)
    d.arrow(xs[2] - 1.6, 12.45, xs[0] + 1.6, 12.45, "resultado de la evaluación", text_dy=-0.24)

    # ---------------- Cierre ----------------
    d.bar(L, R, 13.45, 0.5, "MPI_Reduce(MAX, t_fin) → P0   ·   MPI_Gather(estadísticas: cómputo, "
          "comunicación, reinicios, descartadas) → P0")
    d.box(xs[0], 14.2, 0.6, "Imprime resultados (RESULT)")
    d.bar(L, R, 15.05, 0.45, "MPI_Finalize")

    d.ax.text(L, 15.85, "Dependencia ajuste → evaluación resuelta de forma especulativa: la evaluación "
              "avanza con el mejor K conocido\ny solo es válida la pasada hecha con el K del mensaje FIN. "
              "E se reparte en proporción a la fracción de trabajo estimada para la evaluación.",
              fontsize=FS, color=C_DEP, fontweight="bold", va="top", linespacing=1.35)
    d.legend([("comp", "Cómputo local"), ("spec", "Trabajo especulativo"),
              ("inv", "Invalidación y reinicio"), ("idle", "Espera / ocioso"),
              ("coll", "Colectiva / sincronización"), ("p2p", "Mensaje punto a punto")], 16.45)
    d.save("grafo_mpi_spec")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    diagram_data()
    diagram_spec()
    print("OK")
