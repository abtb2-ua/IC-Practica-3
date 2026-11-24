"""Grafos de la solución paralela MPI (apartado f) como hojas de Excel editables.
Cada caja es una celda o un rango combinado con relleno y borde."""
import sys
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = sys.argv[1] if len(sys.argv) > 1 else "grafos_mpi.xlsx"
FONT = "Arial"

# Paleta (igual que los PNG)
COMP, COMP_E = "DBE8F6", "3B6EA5"
COLL, COLL_E = "F6B26B", "B45F06"
P2P, P2P_E = "D9EAD3", "38761D"
SPEC, SPEC_E = "FFF2CC", "BF9000"
INV, INV_E = "F4CCCC", "990000"
IDLE, IDLE_E = "EEEEEE", "888888"
HEAD = "1F3B57"
LOOP = "555555"
G_TUNE, G_EVAL = "EEF3FA", "FDF6E3"


def side(color, style="thin"):
    return Side(style=style, color=color)


def fill(color, pattern=None, bg=None):
    if pattern:  # trama (trabajo especulativo)
        return PatternFill(fill_type=pattern, fgColor=SPEC_E, bgColor=bg or color)
    return PatternFill(fill_type="solid", fgColor=color)


class Sheet:
    def __init__(self, ws, widths):
        self.ws = ws
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.sheet_view.showGridLines = False

    def cell(self, row, c0, c1=None, text="", fc=None, ec=None, bold=False, color="000000",
             size=9, style="thin", pattern=None, halign="center", italic=False, rotate=0,
             r1=None):
        """Escribe 'text' en el rango (row..r1, c0..c1), lo combina y le pone relleno y borde."""
        c1 = c1 or c0
        r1 = r1 or row
        ws = self.ws
        if (c1, r1) != (c0, row):
            ws.merge_cells(start_row=row, start_column=c0, end_row=r1, end_column=c1)
        top = ws.cell(row=row, column=c0)
        top.value = text
        top.font = Font(name=FONT, size=size, bold=bold, color=color, italic=italic)
        top.alignment = Alignment(horizontal=halign, vertical="center", wrap_text=True,
                                  text_rotation=rotate)
        for r in range(row, r1 + 1):
            for c in range(c0, c1 + 1):
                x = ws.cell(row=r, column=c)
                if fc:
                    x.fill = fill(fc, pattern)
                if ec:
                    s = side(ec, style)
                    x.border = Border(left=s if c == c0 else None, right=s if c == c1 else None,
                                      top=s if r == row else None, bottom=s if r == r1 else None)

    def frame(self, r0, r1, c0, c1, color=LOOP):
        """Marco discontinuo de un bucle (solo el contorno, conserva los bordes internos)."""
        s = side(color, "mediumDashed")
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not (r in (r0, r1) or c in (c0, c1)):
                    continue
                x = self.ws.cell(row=r, column=c)
                b = x.border
                x.border = Border(left=s if c == c0 else b.left, right=s if c == c1 else b.right,
                                  top=s if r == r0 else b.top, bottom=s if r == r1 else b.bottom)

    def band(self, r0, r1, c0, c1, color):
        """Fondo de grupo en las celdas vacías del rango."""
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                x = self.ws.cell(row=r, column=c)
                if x.fill is None or x.fill.fill_type is None:
                    x.fill = fill(color)

    def height(self, row, h):
        self.ws.row_dimensions[row].height = h

    def legend(self, row, col, items):
        r = row
        self.cell(r, col, col + 2, "Leyenda", bold=True, halign="left", size=10)
        for fc, ec, txt, pattern, style in items:
            r += 1
            self.height(r, 18)
            self.cell(r, col, col, "", fc=fc, ec=ec, pattern=pattern, style=style)
            self.cell(r, col + 1, col + 3, txt, halign="left")


def comp(s, r, c, t, **k):
    s.cell(r, c, text=t, fc=COMP, ec=COMP_E, **k)


def coll(s, r, c0, c1, t):
    s.cell(r, c0, c1, t, fc=COLL, ec=COLL_E, bold=True, style="medium")


def p2p(s, r, c, t):
    s.cell(r, c, text=t, fc=P2P, ec=P2P_E)


def arrow(s, r, c0, c1, t):
    s.cell(r, c0, c1, t, bold=True, color=P2P_E, size=9)


# ---------------------------------------------------------------------------
def sheet_data(wb):
    ws = wb.active
    ws.title = "Modo data"
    s = Sheet(ws, [16, 30, 5, 30, 5, 30, 3])
    B, D, F = 2, 4, 6
    s.cell(1, 1, 6, "Solución paralela MPI · modo data (paralelismo de datos, SPMD)",
           bold=True, size=13, halign="left")
    s.height(1, 24)
    for c, t in ((B, "P0\n(raíz)"), (D, "Pr\n(r = 1 … P−2)"), (F, "P(P−1)")):
        s.cell(3, c, text=t, fc=HEAD, ec=HEAD, bold=True, color="FFFFFF")
    for c in (3, 5):
        s.cell(3, c, text="···", bold=True, size=12)
    s.height(3, 32)

    coll(s, 4, B, F, "MPI_Init · MPI_Barrier  →  t0 = MPI_Wtime()   (sincronización inicial)")
    for c in (B, D, F):
        comp(s, 5, c, "Carga MNIST (60 000 train + nTest)\nreplicada en cada proceso")
    s.height(5, 32)
    s.cell(4, 1, 1, "INICIO", bold=True, color="444444", r1=5, fc="F3F3F3", ec="999999")

    s.cell(7, B, F, "loop  K = 1 … maxK   ×   fold = 0 … folds−1", fc=LOOP, color="FFFFFF",
           bold=True, halign="left")
    for c in (B, D, F):
        comp(s, 8, c, "splitFold + KNN(train fold)\n(replicado, sin comunicar)")
    comp(s, 9, B, "Clasifica validación\n[0 , n/P)")
    comp(s, 9, D, "Clasifica validación\n[n·r/P , n·(r+1)/P)")
    comp(s, 9, F, "Clasifica validación\n[n·(P−1)/P , n)")
    coll(s, 10, B, F, "MPI_Allreduce(SUM, aciertos)  ·  sincronización por fold  (maxK·folds veces)")
    for c in (B, D, F):
        comp(s, 11, c, "acc(K) idéntica en todos → actualiza bestK\n(local, sin comunicar)")
    for r in (8, 9):
        s.height(r, 32)
    s.height(11, 42)
    s.height(10, 22)
    s.frame(7, 11, B, F)
    s.cell(7, 1, 1, "FASE 1\nAjuste de K", bold=True, color="444444", r1=11, fc="F3F3F3",
           ec="999999")

    s.cell(13, B, F, "Dependencia de datos: la evaluación necesita el bestK definitivo",
           bold=True, color="CC0000", ec="CC0000", style="mediumDashed")
    comp(s, 14, B, "KNN(train, bestK)\nClasifica test [0 , m/P)")
    comp(s, 14, D, "KNN(train, bestK)\nClasifica test [m·r/P , m·(r+1)/P)")
    comp(s, 14, F, "KNN(train, bestK)\nClasifica test [m·(P−1)/P , m)")
    s.height(14, 32)
    coll(s, 15, B, F, "MPI_Reduce(SUM, aciertos) → P0")
    s.cell(14, 1, 1, "FASE 2\nEvaluación", bold=True, color="444444", r1=15, fc="F3F3F3",
           ec="999999")

    coll(s, 17, B, F, "MPI_Reduce(MAX, t_fin) → P0  ·  MPI_Gather(estadísticas) → P0")
    comp(s, 18, B, "Imprime resultados (línea RESULT)")
    coll(s, 19, B, F, "MPI_Finalize")
    s.cell(17, 1, 1, "RECOGIDA", bold=True, color="444444", r1=19, fc="F3F3F3", ec="999999")
    for r in (4, 10, 15, 17, 19):
        s.height(r, 22)

    s.cell(21, B, F, "n = tamaño del fold de validación (nTune/folds) · m = nTest · "
           "reparto estático por bloques contiguos", halign="left", color="444444", italic=True)
    s.legend(23, B, [(COMP, COMP_E, "Cómputo local", None, "thin"),
                     (COLL, COLL_E, "Comunicación colectiva / sincronización", None, "medium"),
                     (None, "CC0000", "Dependencia entre fases", None, "mediumDashed")])


# ---------------------------------------------------------------------------
def sheet_spec(wb):
    ws = wb.create_sheet("Modo spec")
    s = Sheet(ws, [2, 30, 5, 26, 24, 30, 5, 30, 2])
    B, D, F, H = 2, 4, 6, 8
    s.cell(1, B, H, "Solución paralela MPI · modo spec (paralelismo funcional + ejecución "
           "especulativa)", bold=True, size=13, halign="left")
    s.height(1, 24)
    heads = ((B, "P0\n(raíz global y del ajuste)"), (D, "P(T−1)"),
             (F, "P(T)\n(raíz de la evaluación)"), (H, "P(P−1)"))
    for c, t in heads:
        s.cell(3, c, text=t, fc=HEAD, ec=HEAD, bold=True, color="FFFFFF")
    for c in (3, 7):
        s.cell(3, c, text="···", bold=True, size=12)
    s.height(3, 32)

    coll(s, 4, B, H, "MPI_Init · MPI_Barrier  →  t0 = MPI_Wtime()")
    for c in (B, D, F, H):
        comp(s, 5, c, "Carga MNIST (replicada)")
    coll(s, 6, B, H, "MPI_Comm_split(color = ajuste | evaluación)  →  comunicador 'group'")

    s.cell(7, B, D, "Grupo de ajuste  (T = P − E procesos)", bold=True, color=COMP_E,
           fc=G_TUNE)
    s.cell(7, F, H, "Grupo de evaluación  (E procesos)", bold=True, color=SPEC_E, fc=G_EVAL)

    # Ajuste
    s.cell(8, B, D, "loop  K en orden asc | desc", fc=LOOP, color="FFFFFF", bold=True,
           halign="left")
    for c in (B, D):
        comp(s, 9, c, "CV de K repartida en el grupo:\nsplitFold + KNN y clasifica\nsu bloque de validación")
    coll(s, 10, B, D, "MPI_Allreduce(SUM) en 'group'  (× folds)")
    for c in (B, D):
        comp(s, 11, c, "¿acc(K) mejora?  sí → bestK = K")
    p2p(s, 12, B, "si mejora: MPI_Send NEWK(K)\na cada proceso de evaluación")
    arrow(s, 12, D, 5, "————— NEWK(K) · asíncrono —————→\n(se atiende en el siguiente MPI_Iprobe)")
    s.frame(8, 12, B, D)
    p2p(s, 13, B, "MPI_Send FIN(bestK)\na cada proceso de evaluación")
    arrow(s, 13, D, 5, "————— FIN(bestK definitivo) —————→")
    s.cell(14, B, text="Espera el resultado de la evaluación", fc=IDLE, ec=IDLE_E,
           style="dashed", italic=True)
    s.cell(14, D, text="Termina su parte (ocioso hasta el cierre)", fc=IDLE, ec=IDLE_E,
           style="dashed", italic=True, r1=15)
    p2p(s, 16, B, "MPI_Recv(resultado)")

    # Evaluación
    for c in (F, H):
        s.cell(8, c, text="K = primer K del orden\nKNN(train completo, K)", fc=SPEC,
               ec=SPEC_E, pattern="lightUp")
    s.cell(9, F, H, "loop  hasta FIN y bloque completo", fc=LOOP, color="FFFFFF", bold=True,
           halign="left")
    for c in (F, H):
        p2p(s, 10, c, "MPI_Iprobe + MPI_Recv (no bloqueantes)\ncada 'chunk' muestras")
        s.cell(11, c, text="¿llegó K' ≠ K?\nsí → descarta su bloque y reinicia con K'",
               fc=INV, ec=INV_E)
        s.cell(12, c, text="Clasifica 'chunk' muestras de su bloque\nde test (ESPECULATIVO)",
               fc=SPEC, ec=SPEC_E, pattern="lightUp", bold=True)
        comp(s, 13, c, "Bloque completo: ¿FIN recibido?\nsí → sale · no → MPI_Recv bloqueante")
    s.frame(9, 13, F, H)
    coll(s, 15, F, H, "MPI_Reduce(SUM, aciertos) en 'group' → P(T)")
    p2p(s, 16, F, "MPI_Send(precisión, t_eval) → P0")
    arrow(s, 16, 3, 5, "←—————  resultado de la evaluación  —————")

    coll(s, 18, B, H, "MPI_Reduce(MAX, t_fin) → P0  ·  MPI_Gather(estadísticas: cómputo, "
         "comunicación, reinicios, descartadas) → P0")
    comp(s, 19, B, "Imprime resultados (RESULT)")
    coll(s, 20, B, H, "MPI_Finalize")

    s.band(8, 16, B, D, G_TUNE)
    s.band(8, 16, F, H, G_EVAL)
    for r in (5, 4, 6, 7, 8, 15, 18, 19, 20):
        s.height(r, 22)
    s.height(10, 32)
    for r in (9, 11, 12, 13):
        s.height(r, 40)
    for r in (8, 14, 16):
        s.height(r, 32)
    s.height(12, 44)

    s.cell(22, B, H, "Dependencia ajuste → evaluación resuelta de forma especulativa: la "
           "evaluación avanza con el mejor K conocido y solo es válida la pasada hecha con el K "
           "del mensaje FIN. E se reparte en proporción a la fracción de trabajo estimada para "
           "la evaluación.", bold=True, color="CC0000", halign="left")
    s.height(22, 32)
    s.legend(24, B, [(COMP, COMP_E, "Cómputo local", None, "thin"),
                     (SPEC, SPEC_E, "Trabajo especulativo (trama)", "lightUp", "thin"),
                     (INV, INV_E, "Invalidación y reinicio", None, "thin"),
                     (IDLE, IDLE_E, "Espera / ocioso", None, "dashed"),
                     (COLL, COLL_E, "Colectiva / sincronización", None, "medium"),
                     (P2P, P2P_E, "Mensaje punto a punto (MPI_Send / MPI_Recv / MPI_Iprobe)",
                      None, "thin")])


wb = Workbook()
sheet_data(wb)
sheet_spec(wb)
for ws in wb.worksheets:
    ws.page_setup.orientation = "portrait"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True
wb.save(OUT)
print("OK", OUT)
